from __future__ import annotations
import json, csv, logging, random
from pathlib import Path
from typing import Optional, List, Tuple, Any

logger = logging.getLogger("train_yuuki.dataset")

INSTRUCTION_FORMATS = [
    {"instruction": "instruction", "input": "input",    "output": "output"},
    {"instruction": "prompt",      "input": None,        "output": "response"},
    {"instruction": "question",    "input": None,        "output": "answer"},
    {"instruction": "system",      "input": "user",      "output": "assistant"},
    {"instruction": "human",       "input": None,        "output": "gpt"},
    {"instruction": "from",        "input": None,        "output": "value"},
]

CHAT_ROLES = {"user", "human", "assistant", "gpt", "system", "bot", "ai", "model"}


def detect_format(example: dict) -> str:
    keys = set(k.lower() for k in example.keys())

    if "conversations" in keys or "messages" in keys:
        return "chat_list"

    for fmt in INSTRUCTION_FORMATS:
        if fmt["instruction"] in keys and fmt["output"] in keys:
            return "instruction"

    for k in keys:
        val = example.get(k, "")
        if isinstance(val, list) and val:
            item = val[0]
            if isinstance(item, dict):
                item_keys = set(item.keys())
                if item_keys & CHAT_ROLES or ("role" in item_keys and "content" in item_keys):
                    return "chat_list"

    if "text" in keys or "content" in keys or "code" in keys:
        return "plain"

    return "plain"


def format_as_instruction(example: dict, fmt_type: str) -> str:
    if fmt_type == "chat_list":
        conv_key = "conversations" if "conversations" in example else "messages"
        items = example.get(conv_key, [])
        if not isinstance(items, list):
            return ""
        parts = []
        for item in items:
            if isinstance(item, dict):
                role    = item.get("role", item.get("from", "")).lower()
                content = item.get("content", item.get("value", item.get("text", "")))
                if role in ("user", "human"):
                    parts.append(f"### Human:\n{content}")
                elif role in ("assistant", "gpt", "ai", "model", "bot"):
                    parts.append(f"### Assistant:\n{content}")
                elif role == "system":
                    parts.append(f"### System:\n{content}")
                else:
                    parts.append(str(content))
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)

    if fmt_type == "instruction":
        for fmt in INSTRUCTION_FORMATS:
            inst_key = fmt["instruction"]
            inp_key  = fmt["input"]
            out_key  = fmt["output"]
            if inst_key in example and out_key in example:
                instruction = example.get(inst_key, "")
                inp         = example.get(inp_key, "") if inp_key else ""
                output      = example.get(out_key, "")
                if inp and inp.strip():
                    return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    for key in ["text","content","code","prompt","input","question","answer"]:
        if key in example:
            val = example[key]
            if isinstance(val, str):
                return val.strip()

    parts = []
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n".join(parts)


def flatten_example(example: dict) -> str:
    fmt = detect_format(example)
    return format_as_instruction(example, fmt)


def detect_col(features: dict) -> str:
    priority = ["text","content","code","conversation","conversations","messages",
                "prompt","input","instruction","response","message","question","answer"]
    for col in priority:
        if col in features:
            return col
    for col in features:
        dtype = str(getattr(features.get(col, object()), "dtype", "")).lower()
        if "string" in dtype:
            return col
    return list(features.keys())[0] if features else "text"


def load_local_file(path: str):
    from datasets import Dataset as HFDataset
    ext  = Path(path).suffix.lower()
    rows = []

    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif ext in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
                rows = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            pass
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return HFDataset.from_list(rows)


def preview_dataset(dataset, n: int = 3) -> List[str]:
    previews = []
    count = 0
    for example in dataset:
        if count >= n:
            break
        text = flatten_example(example)
        if text:
            preview = text[:400] + "..." if len(text) > 400 else text
            previews.append(preview)
            count += 1
    return previews


def build_dataset(ds_cfg, tokenizer, max_len: int, val_ratio: float = 0.0):
    from datasets import interleave_datasets, load_dataset

    streams, statics = [], []

    for ds in ds_cfg.datasets:
        try:
            if ds.type == "hf":
                logger.info(f"Loading {ds.label} (streaming)...")
                d = load_dataset(ds.id, ds.config, split=ds.split,
                                 streaming=True, trust_remote_code=True)
                streams.append(d)
            elif ds.type == "local":
                logger.info(f"Loading local: {ds.path}")
                statics.append(load_local_file(ds.path))
        except Exception as e:
            logger.warning(f"Could not load {ds.label}: {e}")

    if not streams and not statics:
        raise RuntimeError("No datasets loaded successfully.")

    cs = None
    if streams:
        cs = (interleave_datasets(streams, stopping_strategy="all_exhausted")
              if len(streams) > 1 else streams[0])
        cs = cs.shuffle(buffer_size=ds_cfg.buffer, seed=42)

    sample = next(iter(cs if cs else statics[0]))
    fmt_type = detect_format(sample)
    logger.info(f"Detected dataset format: {fmt_type}")

    def tokenize(example):
        text = flatten_example(example)
        if not text or not text.strip():
            return {"input_ids": [], "attention_mask": [], "labels": []}
        out = tokenizer(text, truncation=True, max_length=max_len, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    parts = []
    if cs:
        all_cols = list(sample.keys())
        t = cs.map(tokenize, remove_columns=all_cols)
        parts.append(t.filter(lambda x: len(x["input_ids"]) > 0))

    for sd in statics:
        t = sd.map(tokenize, remove_columns=sd.column_names)
        parts.append(t.filter(lambda x: len(x["input_ids"]) > 0))

    if len(parts) == 1:
        combined = parts[0]
    else:
        combined = interleave_datasets(parts, stopping_strategy="all_exhausted")

    return combined, fmt_type


def extend_tokenizer_vocabulary(tokenizer, datasets: list, min_freq: int = 10,
                                 max_new_tokens: int = 2000) -> Tuple[Any, int]:
    logger.info("Analyzing vocabulary for extension...")
    freq: dict = {}
    count = 0

    for ds in datasets:
        for example in ds:
            text = flatten_example(example) if isinstance(example, dict) else str(example)
            words = text.split()
            for word in words:
                freq[word] = freq.get(word, 0) + 1
            count += 1
            if count >= 50000:
                break

    current_vocab = set(tokenizer.get_vocab().keys())
    new_tokens = [
        word for word, cnt in freq.items()
        if cnt >= min_freq and word not in current_vocab and word.isalpha()
    ]
    new_tokens = new_tokens[:max_new_tokens]

    if new_tokens:
        added = tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {added} new tokens to vocabulary.")
        return tokenizer, added

    return tokenizer, 0


def build_dpo_dataset(dataset_id: str, tokenizer, cfg):
    from datasets import load_dataset
    logger.info(f"Loading DPO dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", trust_remote_code=True)

    prompt_col  = next((c for c in ["prompt","question","instruction"] if c in ds.column_names), None)
    chosen_col  = next((c for c in ["chosen","accepted","good","positive"] if c in ds.column_names), None)
    rejected_col = next((c for c in ["rejected","bad","negative"] if c in ds.column_names), None)

    if not all([prompt_col, chosen_col, rejected_col]):
        raise ValueError(
            f"DPO dataset must have prompt, chosen, rejected columns. "
            f"Found: {ds.column_names}"
        )

    def process(example):
        return {
            "prompt":   example[prompt_col],
            "chosen":   example[chosen_col],
            "rejected": example[rejected_col],
        }

    return ds.map(process, remove_columns=ds.column_names)
