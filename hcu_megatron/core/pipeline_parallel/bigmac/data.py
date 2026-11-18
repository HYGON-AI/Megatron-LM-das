import copy
import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace


def _parse_dataset_config(args):
    from hcu_megatron.core.datasets import vlm_args

    if not args.vlm_data_config_path:
        raise ValueError("--vlm-data-config-path must be specified for VLM SFT training")
    config_path = Path(args.vlm_data_config_path)
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    path_base = config.get("path_base")
    if path_base not in (None, "config"):
        raise ValueError(f"Unsupported VLM config path_base: {path_base!r}")
    if "train_data_infos" not in config:
        raise ValueError(f"'train_data_infos' not found in {args.vlm_data_config_path}")

    def parse_split(name, require_probability):
        split = config.get(name)
        if path_base == "config" and split:
            base = config_path.resolve().parent
            split = {key: dict(values) for key, values in split.items()}
            for values in split.values():
                if "path" in values and not Path(values["path"]).is_absolute():
                    values["path"] = (base / values["path"]).resolve().as_posix()
        return vlm_args._parse_split(
            split, require_probability=require_probability, split_name=name,
        )

    train = parse_split("train_data_infos", True)
    assert train is not None
    args.data_path = train.paths
    args.vlm_domain_probabilities = train.probabilities
    args.vlm_train_data_domain_names = train.domain_names
    evaluation = parse_split("eval_data_infos", False) if args.eval_iters > 0 else None
    args.vlm_eval_data_path = evaluation.paths if evaluation is not None else None
    args.vlm_eval_data_domain_names = evaluation.domain_names if evaluation is not None else None
    vlm_args.print_rank_0(
        f"parse_dataset_config: data_path={args.data_path} "
        f"vlm_domain_probabilities={args.vlm_domain_probabilities} "
        f"train_data_domain_names={args.vlm_train_data_domain_names} "
        f"vlm_eval_data_path={args.vlm_eval_data_path}"
    )


def _iter_domain_files_wrapper(original):
    @wraps(original)
    def iterate(self, domain_id):
        reader = copy.copy(self)
        reader.files_by_domain = list(self.files_by_domain)
        for path in self.files_by_domain[domain_id]:
            reader.files_by_domain[domain_id] = [path]
            directory = Path(path).resolve().parent
            for record_number, record in enumerate(original(reader, domain_id), 1):
                path_base = record.pop("image_path_base", None)
                if path_base not in (None, "jsonl"):
                    raise ValueError(
                        f"{path}: record {record_number}: unsupported image_path_base {path_base!r}"
                    )
                if path_base == "jsonl":
                    for image in record.get("images", []):
                        if "tar_name" not in image and not Path(image["image_path"]).is_absolute():
                            image["image_path"] = (directory / image["image_path"]).resolve().as_posix()
                yield record
    return iterate


def _multimodal_iter(self):
    from hcu_megatron.core.datasets.mm_dataset import convert_conversations, fetch_images

    domain_states = SimpleNamespace(domain_lines=0)
    for example in self.underlying:
        json_data = example["json_data"]
        images = None
        if "images" in json_data and len(json_data["images"]) > 0:
            try:
                images = fetch_images(json_data["images"], self.tar_dir)
            except (FileNotFoundError, PermissionError) as error:
                raise RuntimeError(
                    f"VLM image load failed at dp-rank:{self.underlying.dp_rank}: {error}. "
                    "Check images[].image_path inside the JSONL and dataset mounts; "
                    "changing the config path does not rewrite absolute image paths."
                ) from error
            except Exception as error:
                domain_states.domain_lines += example["domain_line"]
                print(
                    f"Abort Sample at dp-rank:{self.underlying.dp_rank}"
                    f"[image load failed: {error}]"
                )
                continue
            if not self._images_are_valid(images):
                domain_states.domain_lines += example["domain_line"]
                print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[invalid image]")
                continue
        conversations = convert_conversations(json_data["conversations"])
        tools = json_data.get("tools", None)
        assert len(conversations) > 1, "SFT 至少需要一轮对话 (user + assistant)"
        images = self._pre_convert_hook(images, json_data)
        del example["json_data"]
        example = self.convert_example(
            copy.deepcopy(example), conversations, images, domain_states, tools,
        )
        if example is not None:
            yield example


def _data_iter_wrapper(original):
    @wraps(original)
    def build(args, *positional, **kwargs):
        if args.num_workers == 0:
            args = copy.copy(args)
            args.vlm_dataloader_prefetch_factor = None
        return original(args, *positional, **kwargs)
    return build


def register_patches(manager):
    prefix = "hcu_megatron.core.datasets."
    manager.register_patch(prefix + "vlm_args.parse_dataset_config", _parse_dataset_config)
    manager.register_patch(
        prefix + "indexed_jsonl_dataset.BaseIndexedJsonlDataset._iter_domain_files",
        _iter_domain_files_wrapper, apply_wrapper=True,
    )
    manager.register_patch(prefix + "mm_dataset.MultiModalDataset.__iter__", _multimodal_iter)
    manager.register_patch(
        prefix + "vlm_dataset.build_train_valid_test_data_iter",
        _data_iter_wrapper, apply_wrapper=True,
    )
