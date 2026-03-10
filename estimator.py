from __future__ import annotations
import math
from typing import Optional, Tuple


KNOWN_MODELS = {
    "yuuki-nxg":    {"params": 3e9,   "layers": 32},
    "yuuki-nano":   {"params": 81e6,  "layers": 12},
    "llama-3-8b":   {"params": 8e9,   "layers": 32},
    "llama-3-70b":  {"params": 70e9,  "layers": 80},
    "mistral-7b":   {"params": 7e9,   "layers": 32},
    "phi-3-mini":   {"params": 3.8e9, "layers": 32},
    "gemma-2b":     {"params": 2e9,   "layers": 18},
    "gemma-7b":     {"params": 7e9,   "layers": 28},
}

DEVICE_THROUGHPUT = {
    "a100-80gb":   {"tokens_per_sec": 55000, "vram_gb": 80,  "name": "A100 80GB"},
    "a100-40gb":   {"tokens_per_sec": 45000, "vram_gb": 40,  "name": "A100 40GB"},
    "h100":        {"tokens_per_sec": 90000, "vram_gb": 80,  "name": "H100 80GB"},
    "rtx4090":     {"tokens_per_sec": 25000, "vram_gb": 24,  "name": "RTX 4090"},
    "rtx3090":     {"tokens_per_sec": 18000, "vram_gb": 24,  "name": "RTX 3090"},
    "rtx3080":     {"tokens_per_sec": 14000, "vram_gb": 10,  "name": "RTX 3080"},
    "rtx3070":     {"tokens_per_sec": 10000, "vram_gb": 8,   "name": "RTX 3070"},
    "rtx4060":     {"tokens_per_sec": 12000, "vram_gb": 8,   "name": "RTX 4060"},
    "v100":        {"tokens_per_sec": 22000, "vram_gb": 16,  "name": "V100 16GB"},
    "t4":          {"tokens_per_sec": 8000,  "vram_gb": 16,  "name": "T4 16GB"},
    "mps":         {"tokens_per_sec": 5000,  "vram_gb": 16,  "name": "Apple MPS"},
    "cpu":         {"tokens_per_sec": 800,   "vram_gb": 0,   "name": "CPU"},
}


def detect_device_key() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        name = torch.cuda.get_device_name(0).lower()
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9)
        if "h100" in name: return "h100"
        if "a100" in name: return "a100-80gb" if vram >= 70 else "a100-40gb"
        if "4090" in name: return "rtx4090"
        if "3090" in name: return "rtx3090"
        if "3080" in name: return "rtx3080"
        if "3070" in name: return "rtx3070"
        if "4060" in name: return "rtx4060"
        if "v100" in name: return "v100"
        if "t4"   in name: return "t4"
        if vram >= 70: return "a100-80gb"
        if vram >= 20: return "rtx3090"
        if vram >= 10: return "rtx3080"
        return "rtx3070"
    except Exception:
        return "cpu"


def estimate_vram_gb(
    param_count: float,
    mode: str,
    precision: str,
    batch_size: int,
    seq_len: int,
    grad_accum: int = 1,
) -> Tuple[float, float]:
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    bpp = bytes_per_param.get(precision, 2)

    if mode == "qlora":
        base_vram = (param_count * 0.5) / 1e9
        adapter_params = param_count * 0.01
        adapter_vram = (adapter_params * 4) / 1e9
        optimizer_vram = (adapter_params * 8) / 1e9
    elif mode == "lora":
        base_vram = (param_count * bpp) / 1e9
        adapter_params = param_count * 0.01
        adapter_vram = (adapter_params * 4) / 1e9
        optimizer_vram = (adapter_params * 8) / 1e9
    else:
        base_vram = (param_count * bpp) / 1e9
        adapter_params = 0
        adapter_vram = 0
        optimizer_vram = (param_count * 8) / 1e9

    activation_vram = (batch_size * seq_len * 4096 * bpp * 4) / 1e9

    total = base_vram + adapter_vram + optimizer_vram + activation_vram
    minimum = base_vram + activation_vram * 0.5

    return round(total, 1), round(minimum, 1)


def estimate_training_time(
    param_count: float,
    mode: str,
    dataset_tokens: int,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    epochs: int,
    max_steps: int,
    device_key: str,
) -> dict:
    dev = DEVICE_THROUGHPUT.get(device_key, DEVICE_THROUGHPUT["cpu"])
    tps = dev["tokens_per_sec"]

    effective_batch = batch_size * seq_len
    total_tokens = dataset_tokens * epochs

    if max_steps > 0:
        total_tokens = min(total_tokens, max_steps * effective_batch * grad_accum)

    if mode == "full":
        tps_adj = tps * 0.6
    elif mode == "lora":
        tps_adj = tps * 0.85
    elif mode == "qlora":
        tps_adj = tps * 0.55
    else:
        tps_adj = tps

    slowdown = math.log10(max(param_count / 1e9, 0.1) + 1) * 0.3 + 0.7
    tps_final = tps_adj * slowdown

    total_seconds = total_tokens / max(tps_final, 1)
    steps = total_tokens // (effective_batch * grad_accum)

    h, r = divmod(int(total_seconds), 3600)
    m, s = divmod(r, 60)

    cost_per_hour = {
        "a100-80gb": 3.20, "a100-40gb": 2.10, "h100": 5.50,
        "rtx4090": 0.80, "rtx3090": 0.70, "v100": 2.50, "t4": 0.35,
        "mps": 0.0, "cpu": 0.0,
    }.get(device_key, 0.0)
    estimated_cost = (total_seconds / 3600) * cost_per_hour

    return {
        "device": dev["name"],
        "total_tokens": total_tokens,
        "estimated_steps": steps,
        "seconds": total_seconds,
        "human": f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s",
        "tokens_per_sec": int(tps_final),
        "estimated_cost_usd": round(estimated_cost, 2),
        "cost_currency": "USD (cloud estimate)" if cost_per_hour > 0 else "free (local)",
    }


def estimate_dataset_tokens(datasets: list, seq_len: int) -> int:
    total = 0
    for ds in datasets:
        if ds.get("type") == "hf":
            total += 500_000 * seq_len
        elif ds.get("type") == "local":
            try:
                import os
                size = os.path.getsize(ds["path"])
                total += (size // 4) * seq_len
            except Exception:
                total += 100_000 * seq_len
    return max(total, 10_000)


def build_estimate_report(cfg) -> str:
    device_key = detect_device_key()
    param_count = 3e9

    try:
        mp = (cfg.model.model_path or "").lower()
        for k, v in KNOWN_MODELS.items():
            if k in mp:
                param_count = v["params"]
                break
    except Exception:
        pass

    ds_list = [{"type": d.type, "path": getattr(d, "path", None)} for d in cfg.dataset.datasets]
    tokens = estimate_dataset_tokens(ds_list, cfg.hp.max_length)

    time_est = estimate_training_time(
        param_count=param_count,
        mode=cfg.hw.mode,
        dataset_tokens=tokens,
        batch_size=cfg.hp.batch_size,
        grad_accum=cfg.hp.grad_accum,
        seq_len=cfg.hp.max_length,
        epochs=cfg.hp.epochs,
        max_steps=cfg.hp.max_steps,
        device_key=device_key,
    )

    vram_total, vram_min = estimate_vram_gb(
        param_count=param_count,
        mode=cfg.hw.mode,
        precision=cfg.hw.precision if cfg.hw.use_gpu else "fp32",
        batch_size=cfg.hp.batch_size,
        seq_len=cfg.hp.max_length,
        grad_accum=cfg.hp.grad_accum,
    )

    lines = [
        f"  Device detected   : {time_est['device']}",
        f"  Estimated time    : {time_est['human']}",
        f"  Estimated steps   : {time_est['estimated_steps']:,}",
        f"  Tokens/sec        : ~{time_est['tokens_per_sec']:,}",
        f"  VRAM needed       : ~{vram_total} GB  (min {vram_min} GB)",
        f"  Cloud cost est.   : {time_est['estimated_cost_usd']} {time_est['cost_currency']}",
    ]
    return "\n".join(lines)
