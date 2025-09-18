# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import os

import hydra
import numpy as np
import ray
from tabulate import tabulate

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import json
import time

import pandas as pd
from transformers import AutoTokenizer

from verl import DataProto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def eval_score(config, dataset, select_reward_fn):
    # 后续进行评价指标的计算（与原实现相同）
    output_dir = os.path.dirname(config.data.output_path)
    os.makedirs(output_dir, exist_ok=True)      
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    saved_data = {}

    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        # score_lst = [reward_fn(prompt[0]['content']+r if config.reward_model.reward_manager == 'deepscaler' else r, ground_truth) for i, r in enumerate(response_lst)]
        score_lst = [reward_fn(r, ground_truth) for i, r in enumerate(response_lst)]
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1
        saved_data[prompt[0]['content']] = score_lst

    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'prompt_score.json'), 'w') as f:
        json.dump(saved_data, f, indent=4)
    return pass_at_n, pass_at_1



@hydra.main(config_path='config', config_name='generation_prior', version_base=None)
def main(config):
    import json
    from pprint import pprint

    import pandas as pd
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # 若存在 output_path，则加载已有数据，否则从原始路径加载，并初始化 responses 列
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} exists, loading existing dataset.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        print(f"Output file does not exist, loading dataset from {config.data.path}.")
        dataset = pd.read_parquet(config.data.path)
        dataset['responses'] = np.array([[""]*config.data.n_samples for _ in range(len(dataset))]).reshape(len(dataset), config.data.n_samples).tolist()

    # 选择 responses 为空的行（为空列表或 NaN）
    n_samples = config.data.n_samples
    missing_mask = dataset['responses'].isna() | dataset['responses'].apply(lambda x: len(x[0]) == 0)
    if missing_mask.sum() != 0:

        indices_to_generate = dataset.index[missing_mask].tolist()
        # 从原 dataset 中获取 prompt 列，并只处理需要生成的部分
        all_prompts = dataset[config.data.prompt_key].tolist()
        chat_lst = [all_prompts[i] for i in indices_to_generate]
        chat_lst = [chat.tolist() if hasattr(chat, 'tolist') else chat for chat in chat_lst]

        from verl.utils import hf_tokenizer
        local_path = copy_local_path_from_hdfs(config.model.path)
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 初始化 Ray worker group
        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(chat_lst)
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + (1 if total_samples % config_batch_size != 0 else 0)

        # 分批生成回答
        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size: (batch_idx+1) * config_batch_size]
            batch_indices = indices_to_generate[batch_idx * config_batch_size: (batch_idx+1) * config_batch_size]

            # 对每个 prompt 重复生成 config.data.n_samples 次
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)

            inputs = tokenizer.apply_chat_template(
                repeated_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors='pt',
                return_dict=True,
                tokenize=True
            )
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]

            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            print(len(data.batch['input_ids']))
            start_time = time.time()
            output = wg.generate_sequences(data)
            print(f'[{batch_idx+1}/{num_batch}] time: {round(time.time()-start_time,1)}.')
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(
                output.batch['input_ids'][:, -config.rollout.response_length:],
                skip_special_tokens=False
            )
            pad_token = tokenizer.pad_token
            output_text_unpad = [text.replace(pad_token, '') for text in output_text]

            # 将输出 reshape 成 (n_data, n_samples)
            num_generated = len(output_text_unpad)
            n_data_generated = num_generated // n_samples
            batch_responses = np.array(output_text_unpad).reshape(n_data_generated, n_samples).tolist()

            # 更新 dataset 中对应行的 responses 字段
            for idx, resp in zip(batch_indices, batch_responses):
                dataset.at[idx, 'responses'] = resp

            # 每个 batch 生成结束后保存 dataset
            output_dir = os.path.dirname(config.data.output_path)
            os.makedirs(output_dir, exist_ok=True)      
            dataset.to_parquet(config.data.output_path)
            print(f"Batch {batch_idx+1} saved to {config.data.output_path}。")

            # 后续进行评价指标的计算（与原实现相同）
            pass_at_n, pass_at_1 = eval_score(config, dataset, select_reward_fn)

    pass_at_n, pass_at_1 = eval_score(config, dataset, select_reward_fn)
    output_dir = os.path.dirname(config.data.output_path)
    all_prompts = dataset['prompt'].tolist()
    all_idx = []
    for extra_info in dataset['extra_info']:
        all_idx.append(extra_info['index'])
    json_path = os.path.join(output_dir, 'prompt_score.json')
    with open(json_path, 'r') as f:
        score_data = json.load(f)

    index_score = {}

    for idx, prompt in zip(all_idx, all_prompts):
        if prompt[0]['content'] in score_data.keys():
            index_score[str(idx)] = score_data[prompt[0]['content']]

    with open(os.path.join(os.path.dirname(json_path), 'index_score.json'), 'w') as f:
        json.dump(index_score, f, indent=4)

    csv_path = os.path.join(output_dir, 'pass.csv')
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    table_data = [[k, v] for k, v in row_data.items()]
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from verl.utils.reward_score import math
        return math.compute_score
    elif data_source == 'countdown':
        from recipe.ours.reward_func.countdown_reward import compute_score
        return compute_score
    elif data_source == '':
        from recipe.ours.reward_func.deepscaler_reward import \
            deepscaler_reward_fn
        return deepscaler_reward_fn

if __name__ == '__main__':
    main()