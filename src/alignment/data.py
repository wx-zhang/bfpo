# coding=utf-8
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
import os
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo", "safe_ipo"],
):
    if task in ["sft", "generation"]:
        if example["dataset_name"] == "HuggingFaceH4/ultrachat_200k":  # Ultra Chat dataset
            messages = example["messages"]
            # We add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})

        elif example['dataset_name'] == "PKU-Alignment/PKU-SafeRLHF":  # pku safe rlhf
            messages = [
                {'role': "system", "content": ""},
                {'role': "user", "content": example['prompt']},
            ]

            messages.append(
                {'role': "assistant", "content": example[f"response_{example['safer_response_id']}"]})

        elif example['dataset_name'] == "snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset":
            messages = example["chosen"]
            # We add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
        else:
            raise NotImplementedError

        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )

    elif task == "bfpo":
        if example["dataset_name"] in ["HuggingFaceH4/ultrafeedback_binarized"]:  # Ultra Chat dataset
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            is_chosen_safe = int(1)
            is_rejected_safe = int(1)

        elif example['dataset_name'] == "PKU-Alignment/PKU-SafeRLHF":  # pku safe rlhf

            chosen_id = example['better_response_id']
            reject_id = int(1-chosen_id)
            chosen_messages = [{'role': "assistant",
                                "content": example[f'response_{chosen_id}']}]
            rejected_messages = [{'role': "assistant",
                                  "content": example[f'response_{reject_id}']}]
            prompt_messages = [
                {'role': "system", "content": ""},
                {'role': "user", "content": example['prompt']},
            ]
            is_chosen_safe = 1 if example[f"is_response_{chosen_id}_safe"] else 0
            is_rejected_safe = 1 if example[f"is_response_{reject_id}_safe"] else 0
        elif example['dataset_name'] == "data/zephyr-7b-beta_beaver_redteaming.json":  # pku safe rlhf

            chosen_id = example['better_modelout']
            reject_id = 3-int(chosen_id)  # the id is 1 or 2
            chosen_messages = [{'role': "assistant",
                                "content": example[f'modeloutput{chosen_id}']}]
            rejected_messages = [
                {'role': "assistant", "content": example[f'modeloutput{reject_id}']}]
            prompt_messages = [
                {'role': "system", "content": ""},
                {'role': "user", "content": example['prompt']},
            ]
            is_chosen_safe = 1 if example[f"is_modelout{chosen_id}_safe"] else 0
            is_rejected_safe = 1 if example[f"is_modelout{reject_id}_safe"] else 0

        else:
            raise ValueError(
                f"Could not format example as dialogue for `bfpo` task!"
            )
        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False)
        example['is_chosen_safe'] = is_chosen_safe
        example['is_rejected_safe'] = is_rejected_safe
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'bfpo']}"
        )
    return example


def add_dataset_name(examples, dataset_name):
    # Add a new field with the dataset name
    examples["dataset_name"] = dataset_name
    return examples


def get_datasets(
    data_config: DataArguments | dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
    buffer: bool = False,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer if not buffer else data_config.buffer_dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


SPLIT_MAPPTING = {
    'HuggingFaceH4/ultrachat_200k': {'train_sft': 'train_sft', 'test_sft': 'test_sft'},
    'PKU-Alignment/PKU-SafeRLHF': {'train_sft': 'train', 'test_sft': 'test', 'train_prefs': 'train', 'test_prefs': 'test'},
    'HuggingFaceH4/ultrafeedback_binarized': {'train_prefs': 'train_prefs', 'test_prefs': 'test_prefs'},
    'data/zephyr-7b-beta_beaver_redteaming.json': {'train': 'train', 'test': 'train'}
}


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds_load, frac in dataset_mixer.items():

        fracs.append(frac)
        for split in splits:
            try:
                if ds_load.endswith('.json'):
                    dataset = load_dataset(
                        'json', data_files=ds_load, split=SPLIT_MAPPTING[ds_load][split] if ds_load in SPLIT_MAPPTING.keys() else split)
                else:
                    # Try first if dataset on a Hub repo
                    if ds_load == "PKU-Alignment/PKU-SafeRLHF":
                        dataset = load_dataset(ds_load, split=SPLIT_MAPPTING[ds_load][split] if ds_load in SPLIT_MAPPTING.keys(
                        ) else split, revision="ff7ba91063016c78a225b0f74e1c0860bb18230f")
                    else:

                        dataset = load_dataset(
                            ds_load, split=SPLIT_MAPPTING[ds_load][split] if ds_load in SPLIT_MAPPTING.keys() else split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds_load, split))
            # Apply the function to each datapoint in the dataset
            dataset = dataset.map(add_dataset_name, fn_kwargs={
                                  'dataset_name': ds_load})

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(
                train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(
                raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
