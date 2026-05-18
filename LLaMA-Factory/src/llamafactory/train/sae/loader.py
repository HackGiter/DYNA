# Copyright 2024 the LlamaFactory team.
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
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk, load_dataset_builder, DownloadMode
from transformers.utils.versions import require_version

from ...extras.constants import FILEEXT2TYPE
from ...extras.logging import get_logger
from ...extras.misc import has_tokenized_data
from ...data.aligner import align_dataset
from ...data.data_utils import merge_dataset, split_dataset
from ...data.parser import get_dataset_list
from ...data.preprocess import get_preprocess_and_print_func

import datasets

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import DataArguments, ModelArguments
    from ...data.data_utils import DatasetModule
    from ...data.parser import DatasetAttr
    from ...data.template import Template

_URL_BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
_LANGUAGES = ("en", "de", "fr", "es", "it")
_MISSING_FILES_PATTERN = "urls/missing-{component}.txt"
_NUM_SHARDS = 5000

_CC_SNAPSHOT_IDS = (
    "2014-15",
    "2014-23",
    "2014-35",
    "2014-41",
    "2014-42",
    "2014-49",
    "2014-52",
    "2015-14",
    "2015-22",
    "2015-27",
    "2015-32",
    "2015-35",
    "2015-40",
    "2015-48",
    "2016-07",
    "2016-18",
    "2016-22",
    "2016-26",
    "2016-30",
    "2016-36",
    "2016-40",
    "2016-44",
    "2016-50",
    "2017-04",
    "2017-09",
    "2017-17",
    "2017-22",
    "2017-26",
    "2017-30",
    "2017-34",
    "2017-39",
    "2017-43",
    "2017-47",
    "2017-51",
    "2018-05",
    "2018-09",
    "2018-13",
    "2018-17",
    "2018-22",
    "2018-26",
    "2018-30",
    "2018-34",
    "2018-39",
    "2018-43",
    "2018-47",
    "2018-51",
    "2019-04",
    "2019-09",
    "2019-13",
    "2019-18",
    "2019-22",
    "2019-26",
    "2019-30",
    "2019-35",
    "2019-39",
    "2019-43",
    "2019-47",
    "2019-51",
    "2020-05",
    "2020-10",
    "2020-16",
    "2020-24",
    "2020-29",
    "2020-34",
    "2020-40",
    "2020-45",
    "2020-50",
    "2021-04",
    "2021-10",
    "2021-17",
    "2021-21",
    "2021-25",
    "2021-31",
    "2021-39",
    "2021-43",
    "2021-49",
    "2022-05",
    "2022-21",
    "2022-27",
    "2022-33",
    "2022-40",
    "2022-49",
    "2023-06",
    "2023-14",
)


logger = get_logger(__name__)

def _split_generators(self, dl_manager):
    snapshots = getattr(self.config, "snapshots", _CC_SNAPSHOT_IDS)
    languages = getattr(self.config, "languages", _LANGUAGES)
    partition = getattr(self.config, "partition", "all")
    num_shards = getattr(self.config, "num_shards", _NUM_SHARDS)

    if partition == "all":
        partitions = ["head", "middle", "tail"]
    elif partition == "head_middle":
        partitions = ["head", "middle"]
    elif partition == "tail":
        partitions = [partition]
    else:
        raise ValueError(f"invalid partition: {partition}")

    # fetch list of missing files (e.g., missing duplicates or corrupted documents and
    # quality signal files)
    missing_files_paths = dl_manager.download_and_extract(
        {
            component: _MISSING_FILES_PATTERN.format(component=component)
            for component in ("documents", "signals", "duplicates")
        }
    )

    missing_files = {}
    for component, missing_file in missing_files_paths.items():
        with open(missing_file, "r", encoding="utf-8") as f:
            missing_files[component] = set(line.strip() for line in f)

    with open("missing.txt", 'w') as f:
        for k, v in missing_files.items():
            for item in v:
                f.write(item+'\n')

    # build list of urls to fetch
    documents_urls = {}
    quality_signals_urls = {}
    duplicates_ids_urls = {}
    base_tags = []

    for lang in languages:
        for snapshot in snapshots:
            for part in partitions:
                for n in range(num_shards):
                    base_tag = f"{snapshot}/{n:04d}/{lang}_{part}"
                    base_tags.append(base_tag)

                    # documents
                    url = f"{_URL_BASE}/documents/{base_tag}.json.gz"
                    if url not in missing_files["documents"]:
                        documents_urls[base_tag] = url
                    if part != "tail":
                        # quality signals
                        url = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
                        if url not in missing_files["signals"]:
                            quality_signals_urls[base_tag] = url

                        # duplicates
                        url = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"
                        if url not in missing_files["duplicates"]:
                            duplicates_ids_urls[base_tag] = url

    # download documents files
    logger.info(f"Downloading {len(documents_urls)} documents files.")
    documents_files = dl_manager.download(documents_urls)

    # download quality signals files
    logger.info(f"Downloading {len(quality_signals_urls)} quality signals files.")
    quality_signals_files = dl_manager.download(quality_signals_urls)

    # download duplicates ids files
    logger.info(f"Downloading {len(duplicates_ids_urls)} duplicates ids files.")
    duplicates_ids_files = dl_manager.download(duplicates_ids_urls)

    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "base_tags": base_tags,
                "documents_files": documents_files,
                "quality_signals_files": quality_signals_files,
                "duplicates_ids_files": duplicates_ids_files,
            },
        )
    ]


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    if dataset_attr.load_from == "ms_hub":
        require_version("modelscope>=1.11.0", "To fix: pip install modelscope>=1.11.0")
        from modelscope import MsDataset
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()
    else:
        # if "RedPajama-Data-V2" in data_path:
        #     snapshots = [
        #         "2014-15", "2014-41",
        #         "2015-14", "2015-32",
        #         "2016-07", "2016-36",
        #         "2017-04", "2017-34",
        #         "2018-05", "2018-26",
        #         "2019-04", "2019-30",
        #         "2020-05", "2020-34",
        #         "2021-04", "2021-25",
        #         "2022-05", "2022-33",
        #         "2023-06", "2023-14",
        #     ]
        #     languages=_LANGUAGES
        #     partition="head_middle"
        #     num_shards = 8
        #     download_mode = DownloadMode(DownloadMode.REUSE_DATASET_IF_EXISTS)
        #     builder = load_dataset_builder(
        #         "togethercomputer/RedPajama-Data-V2",
        #         name="default",
        #         partition=partition,
        #         snapshots=snapshots,
        #         languages=languages,
        #         # cache_dir="/data1/lihz/.cache/huggingface/datasets/RedPajama-Data-V2",
        #         download_mode=download_mode,
        #         data_dir=data_dir,
        #         data_files=data_files,
        #         cache_dir=model_args.cache_dir,
        #         token=model_args.hf_hub_token,
        #         trust_remote_code=True,
        #     )
        #     builder._split_generators = _split_generators.__get__(builder)
        #     builder.config.num_shards = num_shards
        #     builder.download_and_prepare(num_proc=data_args.preprocessing_num_workers, download_mode=download_mode)
        #     if (data_args.streaming and (dataset_attr.load_from != "file")):
        #         dataset = builder.as_streaming_dataset(split=dataset_attr.split)
        #     else:
        #         dataset = builder.as_dataset(split=dataset_attr.split)
        # else:
        #     dataset = load_dataset(
        #         path=data_path,
        #         name=data_name,
        #         data_dir=data_dir,
        #         data_files=data_files,
        #         split=dataset_attr.split,
        #         cache_dir=model_args.cache_dir,
        #         token=model_args.hf_hub_token,
        #         streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        #         trust_remote_code=True,
        #     )
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
            trust_remote_code=True,
        )

    if data_args.streaming and (dataset_attr.load_from == "file"):  # faster than specifying streaming=True
        dataset = dataset.to_iterable_dataset()  # TODO: add num shards parameter

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets.append(_load_single_dataset(dataset_attr, model_args, data_args, training_args))

    return merge_dataset(datasets, data_args, seed=training_args.seed)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func, print_function = get_preprocess_and_print_func(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load tokenized dataset
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning("Loading dataset from disk will ignore other data arguments.")
            dataset_dict: "DatasetDict" = load_from_disk(data_args.tokenized_path)
            logger.info("Loaded tokenized dataset from {}.".format(data_args.tokenized_path))

            dataset_module: Dict[str, "Dataset"] = {}
            if "train" in dataset_dict:
                dataset_module["train_dataset"] = dataset_dict["train"]

            if "validation" in dataset_dict:
                dataset_module["eval_dataset"] = dataset_dict["validation"]

            if data_args.streaming:
                dataset_module = {k: v.to_iterable_dataset() for k, v in dataset_module.items()}

            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
        )

        if data_args.val_size > 1e-6:
            dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        else:
            dataset_dict = {}
            if dataset is not None:
                if data_args.streaming:
                    dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["train"] = dataset

            if eval_dataset is not None:
                if data_args.streaming:
                    eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)

                dataset_dict["validation"] = eval_dataset

            dataset_dict = DatasetDict(dataset_dict)

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info("Tokenized dataset saved at {}.".format(data_args.tokenized_path))
                logger.info("Please restart the training with `tokenized_path: {}`.".format(data_args.tokenized_path))

            sys.exit(0)

        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]

        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]

        return dataset_module
