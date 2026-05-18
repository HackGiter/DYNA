from tqdm import tqdm
from datasets import load_dataset

_DESCRIPTION = """\
RedPajama V2: an Open Dataset for Training Large Language Models
"""

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

# print(len(_CC_SNAPSHOT_IDS))

# snapshots = [
#     "2023-06",
#     "2023-14",
# ]

dataset = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    name="sample-10B",
    cache_dir="/data1/lihz/.cache/huggingface/datasets/RedPajama-Data-V2",
    trust_remote_code=True,
    num_proc=16,
)

# tag=200
# print(type(dataset), dataset.keys())
# print(type(dataset['train']))
# print(dataset['train'][tag].keys())
# # print(dataset['train'][tag]['raw_content'])
# print(dataset['train'][tag]['meta'])
# print(type(dataset['train'][tag]['meta']))
# print(dataset['train'][tag]['quality_signals'])
# print()

# tag=1010
# print(type(dataset), dataset.keys())
# print(type(dataset['train']))
# print(dataset['train'][tag].keys())
# # print(dataset['train'][tag]['raw_content'])

# def str2dict(x:str) -> dict:
#     x = x.replace('{', '').replace('}', '')
#     x = x.split(', "')
#     x = {item.split(': "')[0]:item.split(': "')[1] for item in x}
#     x = {k.replace('"', ''):v.replace('"', '') for k, v in x.items()}
#     return x

# # sample = -1
# # for i, item in enumerate(tqdm(dataset['train'])):
# #     if i == sample:
# #         break
# #     s2d = str2dict(item['meta'])
# #     # print(s2d)
# #     # print(s2d['url'])
# #     if 'github' in s2d['url']:
# #         print(item['raw_content'])

# import datasets
# from datasets import load_dataset_builder, DownloadMode

# logger = datasets.logging.get_logger(__name__)

# def _split_generators(self, dl_manager):
#     snapshots = getattr(self.config, "snapshots", _CC_SNAPSHOT_IDS)
#     languages = getattr(self.config, "languages", _LANGUAGES)
#     partition = getattr(self.config, "partition", "all")
#     num_shards = getattr(self.config, "num_shards", _NUM_SHARDS)

#     if partition == "all":
#         partitions = ["head", "middle", "tail"]
#     elif partition == "head_middle":
#         partitions = ["head", "middle"]
#     elif partition == "tail":
#         partitions = [partition]
#     else:
#         raise ValueError(f"invalid partition: {partition}")

#     # fetch list of missing files (e.g., missing duplicates or corrupted documents and
#     # quality signal files)
#     missing_files_paths = dl_manager.download_and_extract(
#         {
#             component: _MISSING_FILES_PATTERN.format(component=component)
#             for component in ("documents", "signals", "duplicates")
#         }
#     )

#     missing_files = {}
#     for component, missing_file in missing_files_paths.items():
#         with open(missing_file, "r", encoding="utf-8") as f:
#             missing_files[component] = set(line.strip() for line in f)

#     with open("missing.txt", 'w') as f:
#         for k, v in missing_files.items():
#             for item in v:
#                 f.write(item+'\n')

#     # build list of urls to fetch
#     documents_urls = {}
#     quality_signals_urls = {}
#     duplicates_ids_urls = {}
#     base_tags = []

#     for lang in languages:
#         for snapshot in snapshots:
#             for part in partitions:
#                 for n in range(num_shards):
#                     base_tag = f"{snapshot}/{n:04d}/{lang}_{part}"
#                     base_tags.append(base_tag)

#                     # documents
#                     url = f"{_URL_BASE}/documents/{base_tag}.json.gz"
#                     if url not in missing_files["documents"]:
#                         documents_urls[base_tag] = url
#                     if part != "tail":
#                         # quality signals
#                         url = f"{_URL_BASE}/quality_signals/{base_tag}.signals.json.gz"
#                         if url not in missing_files["signals"]:
#                             quality_signals_urls[base_tag] = url

#                         # duplicates
#                         url = f"{_URL_BASE}/duplicates/{base_tag}.duplicates.parquet"
#                         if url not in missing_files["duplicates"]:
#                             duplicates_ids_urls[base_tag] = url

#     # download documents files
#     logger.info(f"Downloading {len(documents_urls)} documents files.")
#     documents_files = dl_manager.download(documents_urls)

#     # download quality signals files
#     logger.info(f"Downloading {len(quality_signals_urls)} quality signals files.")
#     quality_signals_files = dl_manager.download(quality_signals_urls)

#     # download duplicates ids files
#     logger.info(f"Downloading {len(duplicates_ids_urls)} duplicates ids files.")
#     duplicates_ids_files = dl_manager.download(duplicates_ids_urls)

#     return [
#         datasets.SplitGenerator(
#             name=datasets.Split.TRAIN,
#             gen_kwargs={
#                 "base_tags": base_tags,
#                 "documents_files": documents_files,
#                 "quality_signals_files": quality_signals_files,
#                 "duplicates_ids_files": duplicates_ids_files,
#             },
#         )
#     ]

# snapshots = [
#     "2014-15", "2014-41",
#     "2015-14", "2015-32",
#     "2016-07", "2016-36",
#     "2017-04", "2017-34",
#     "2018-05", "2018-26",
#     "2019-04", "2019-30",
#     "2020-05", "2020-34",
#     "2021-04", "2021-25",
#     "2022-05", "2022-33",
#     "2023-06", "2023-14",
# ]
# languages=_LANGUAGES
# partition="head_middle"
# num_shards = 8

# download_mode = DownloadMode(DownloadMode.REUSE_DATASET_IF_EXISTS)
# builder = load_dataset_builder(
#     "togethercomputer/RedPajama-Data-V2",
#     name="default",
#     partition=partition,
#     snapshots=snapshots,
#     languages=languages,
#     cache_dir="/data1/lihz/.cache/huggingface/datasets/RedPajama-Data-V2",
#     download_mode=download_mode
#     )
# builder._split_generators = _split_generators.__get__(builder)
# builder.config.num_shards = num_shards
# builder.download_and_prepare(num_proc=4, download_mode=download_mode)
# dataset = builder.as_dataset(split='train')
# sample = -1
# for i, item in enumerate(tqdm(dataset)):
#     if i == sample:
#         break
#     s2d = str2dict(item['meta'])
#     # print(s2d)
#     # print(s2d['url'])
#     if 'github' in s2d['url']:
#         print(item['raw_content'])