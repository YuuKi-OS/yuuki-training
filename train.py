#!/usr/bin/env python3
"""
Yuuki NxG Training Wizard  —  OpceanAI
Run: python train.py
"""

import os, sys, signal, logging, threading, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from core.config import (
    TrainingConfig, ModelConfig, HardwareConfig, LoraConfig,
    DatasetConfig, DatasetEntry, HyperParams, LoggingConfig,
    WebhookConfig, PostTrainingConfig, DPOConfig, OptimizationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_yuuki")

C  = "\033[96m"
G  = "\033[92m"
Y  = "\033[93m"
R  = "\033[91m"
B  = "\033[1m"
D  = "\033[2m"
RS = "\033[0m"

HAS_TORCH = False
HAS_BNB   = False
HAS_HUB   = False

try:
    import torch;              HAS_TORCH = True
except ImportError: pass
try:
    import bitsandbytes;       HAS_BNB = True
except ImportError: pass
try:
    from huggingface_hub import HfApi, hf_hub_download, login as hf_login
    HAS_HUB = True
except ImportError: pass


def banner():
    print(f"""\n{B}{C}
  ██╗   ██╗██╗   ██╗██╗   ██╗██╗  ██╗██╗
  ╚██╗ ██╔╝╚██╗ ██╔╝██║   ██║██║ ██╔╝██║
   ╚████╔╝  ╚████╔╝ ██║   ██║█████╔╝ ██║
    ╚██╔╝    ╚██╔╝  ██║   ██║██╔═██╗ ██║
     ██║      ██║   ╚██████╔╝██║  ██╗██║
     ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝
       NxG Training Wizard  v2.0  —  OpceanAI{RS}
""")


def sec(title):
    print(f"\n{B}{Y}{'─'*62}{RS}\n{B}{Y}  {title}{RS}\n{B}{Y}{'─'*62}{RS}\n")

def ok(m):    print(f"  {G}✓{RS} {m}")
def warn(m):  print(f"  {Y}⚠{RS} {m}")
def err(m):   print(f"  {R}✗{RS} {m}")
def info(m):  print(f"  {C}ℹ{RS} {m}")

def _hs(b):
    if b < 1024:  return f"{b} B"
    if b < 1<<20: return f"{b/1024:.1f} KB"
    if b < 1<<30: return f"{b/(1<<20):.1f} MB"
    return f"{b/(1<<30):.2f} GB"

def dbar(desc, cur, total, width=38):
    frac   = min(cur/total, 1.0) if total > 0 else 0
    filled = int(width * frac)
    bar    = ":" * filled + " " * (width - filled)
    pct    = f"{frac*100:5.1f}%" if total > 0 else "  ..."
    print(f"\r  {C}{desc:<28}{RS} [{G}{bar}{RS}] {pct}  {_hs(cur)}/{_hs(total) if total>0 else '?'}", end="", flush=True)

def ddone(desc):
    print(f"\r  {C}{desc:<28}{RS} [{G}{'='*38}{RS}] {G}Done ✓{RS}        ")

def ask(prompt, default=None, choices=None):
    sfx = f" {D}[{default}]{RS}" if default is not None else ""
    if choices: sfx += f" {D}({'/'.join(choices)}){RS}"
    while True:
        raw = input(f"  {B}>{RS} {prompt}{sfx}: ").strip()
        if not raw and default is not None: return default
        if choices and raw.lower() not in [c.lower() for c in choices]:
            warn(f"Choose: {', '.join(choices)}"); continue
        if raw: return raw

def ask_int(prompt, default, min_val=1, max_val=None):
    while True:
        raw = input(f"  {B}>{RS} {prompt} {D}[{default}]{RS}: ").strip()
        if not raw: return default
        try:
            v = int(raw)
            if v < min_val: warn(f"Min is {min_val}"); continue
            if max_val and v > max_val: warn(f"Max is {max_val}"); continue
            return v
        except ValueError: warn("Enter a valid integer.")

def ask_float(prompt, default):
    while True:
        raw = input(f"  {B}>{RS} {prompt} {D}[{default}]{RS}: ").strip()
        if not raw: return default
        try: return float(raw)
        except: warn("Enter a valid number.")

def ask_bool(prompt, default=True):
    return ask(prompt, default="y" if default else "n", choices=["y","n"]).lower() == "y"

def menu(title, opts, default="1"):
    print(f"\n  {B}{title}:{RS}")
    for k, label in opts.items():
        arrow = f"{G}→{RS}" if k == default else " "
        print(f"  {arrow} {C}{k}{RS}. {label}")
    return ask("Select", default=default, choices=list(opts.keys()))


def hf_dl_model(repo_id, local_dir):
    if not HAS_HUB: raise RuntimeError("huggingface_hub not installed.")
    ok(f"Downloading {repo_id} → {local_dir}")
    api   = HfApi()
    files = list(api.list_repo_files(repo_id))
    os.makedirs(local_dir, exist_ok=True)
    for fname in files:
        try:
            info_list = api.get_paths_info(repo_id, [fname])
            size = info_list[0].size if info_list else 0
        except: size = 0
        short = (fname[:26]+"..") if len(fname)>28 else fname
        dbar(short, 0, size)
        hf_hub_download(repo_id=repo_id, filename=fname, local_dir=local_dir)
        ddone(short)
    return local_dir

def hf_dl_gguf(repo_id, filename, dest_dir):
    if not HAS_HUB: raise RuntimeError("huggingface_hub not installed.")
    os.makedirs(dest_dir, exist_ok=True)
    short = (filename[:26]+"..") if len(filename)>28 else filename
    dbar(short, 0, 0)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
    ddone(short)
    return path


def w_model() -> ModelConfig:
    sec("1 / 7  — Model Source")

    c = menu("Where is your model?", {
        "1": "HuggingFace ID  (auto-download on first use)",
        "2": "Download from HuggingFace now  (file-by-file progress)",
        "3": "Local directory  (already downloaded)",
        "4": "Local / HF  .gguf  — inference setup only",
        "5": "Local  .pt / .safetensors / .bin  file",
    })

    model_path, gguf_path, is_hf = None, None, True
    existing_lora = None

    if c == "1":
        model_path = ask("HuggingFace repo ID", default="OpceanAI/Yuuki-NxG")
        ok(f"Will use {model_path}")
    elif c == "2":
        repo = ask("HuggingFace repo ID", default="OpceanAI/Yuuki-NxG")
        dest = ask("Download to", default=f"./models/{repo.split('/')[-1]}")
        if os.path.exists(dest) and os.listdir(dest):
            if not ask_bool("Already exists — re-download?", default=False):
                model_path = dest
            else:
                hf_dl_model(repo, dest); model_path = dest
        else:
            hf_dl_model(repo, dest); model_path = dest
    elif c == "3":
        model_path = ask("Path to local model directory")
        if not os.path.isdir(model_path): err(f"Not found: {model_path}"); sys.exit(1)
    elif c == "4":
        src = menu("GGUF source", {"1":"Download from HuggingFace","2":"Local file"})
        if src == "1":
            repo     = ask("HuggingFace repo (e.g. bartowski/Meta-Llama-3-8B-GGUF)")
            fname    = ask("Filename (e.g. Meta-Llama-3-8B-Q4_K_M.gguf)")
            dest_dir = ask("Download to", default="./models/gguf")
            gguf_path = hf_dl_gguf(repo, fname, dest_dir)
        else:
            gguf_path = ask("Path to .gguf file")
            if not os.path.isfile(gguf_path): err(f"Not found: {gguf_path}"); sys.exit(1)
        is_hf = False
        warn("GGUF → inference/Ollama setup only, no training.")
    elif c == "5":
        model_path = ask("Path to file or directory")
        if not os.path.exists(model_path): err(f"Not found: {model_path}"); sys.exit(1)

    if is_hf and model_path:
        if ask_bool("Continue from existing LoRA adapter?", default=False):
            existing_lora = ask("Path to LoRA adapter directory")
            if not os.path.isdir(existing_lora):
                warn("Path not found — ignoring."); existing_lora = None

    return ModelConfig(
        source="local" if c in ("3","4","5") else "hf_id",
        model_path=model_path,
        gguf_path=gguf_path,
        model_is_hf=is_hf,
        load_existing_lora=existing_lora,
    )


def w_hw(is_gguf=False) -> HardwareConfig:
    sec("2 / 7  — Training Mode & Hardware")

    if is_gguf:
        ok("GGUF → skipping training config.")
        return HardwareConfig(mode="none", use_gpu=False)

    cuda_ok = HAS_TORCH and torch.cuda.is_available()
    mps_ok  = HAS_TORCH and hasattr(torch.backends,"mps") and torch.backends.mps.is_available()

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory/1e9,1)
        ok(f"CUDA: {name}  ({vram} GB VRAM)")
        if HAS_BNB: ok("bitsandbytes available — int8/int4 enabled")
        try:
            import flash_attn; ok("flash-attn available — Flash Attention 2 enabled")
        except: warn("flash-attn not found — install for 2x speed: pip install flash-attn")
    elif mps_ok:
        ok("Apple MPS detected.")
    else:
        warn("No GPU — CPU only. Training will be very slow.")

    mode_c = menu("Training mode", {
        "1": "Full fine-tune   — all weights, needs most VRAM",
        "2": "LoRA             — adapters only, fast + low VRAM  ← recommended",
        "3": "QLoRA            — LoRA + 4-bit base, minimum VRAM",
    }, default="2")
    mode = {"1":"full","2":"lora","3":"qlora"}[mode_c]

    use_gpu = False
    precision = "fp32"
    quant = "none"
    flash_attn = False
    compile_m = False
    multi_gpu = False

    if cuda_ok:
        use_gpu = ask_bool("Use GPU", default=True)

    if use_gpu and cuda_ok:
        if mode == "qlora":
            precision, quant = "bf16", "int4"
            ok("QLoRA: auto-set bf16 + int4 NF4.")
        else:
            pc = menu("Precision", {
                "1":"fp16  — fast, standard  ← recommended",
                "2":"bf16  — stable (Ampere+ only)",
                "3":"fp32  — full precision, slow",
            })
            precision = {"1":"fp16","2":"bf16","3":"fp32"}[pc]
            if mode == "full" and HAS_BNB:
                qc = menu("Base model weight quantization", {
                    "1":"none  — full float weights",
                    "2":"int8  — 8-bit (saves ~50% VRAM)",
                }, default="1")
                quant = {"1":"none","2":"int8"}[qc]

        try:
            import flash_attn
            flash_attn = ask_bool("Flash Attention 2  (faster + less VRAM)", default=True)
        except ImportError:
            pass

        compile_m = ask_bool("torch.compile  (PyTorch 2.0+ — faster after warmup)", default=False)

        if torch.cuda.device_count() > 1:
            info(f"{torch.cuda.device_count()} GPUs detected.")
            multi_gpu = ask_bool("Use all GPUs with accelerate?", default=False)

    elif mps_ok:
        use_gpu, precision = True, "fp32"
        warn("MPS: fp32 only.")

    return HardwareConfig(
        mode=mode, use_gpu=use_gpu, precision=precision,
        quantization=quant, flash_attn=flash_attn,
        compile=compile_m, multi_gpu=multi_gpu,
    )


def w_lora(mode: str) -> LoraConfig:
    if mode not in ("lora","qlora"):
        return LoraConfig()
    sec("2b / 7  — LoRA Config")

    r      = ask_int("Rank  (r)  — higher = more capacity, more VRAM", default=16, min_val=1)
    alpha  = ask_int("Alpha  — scaling factor, convention is 2×r", default=r*2, min_val=1)
    drop   = ask_float("Dropout  — regularization (0 = none)", default=0.05)

    print(f"\n  Target modules {D}(Enter = auto-detect  |  e.g. q_proj,v_proj,k_proj,o_proj){RS}")
    raw = input(f"  {B}>{RS} Modules: ").strip()
    modules = [m.strip() for m in raw.split(",")] if raw else None

    merge = False
    if mode == "lora":
        merge = ask_bool("Merge adapters into base model after training", default=False)

    return LoraConfig(r=r, alpha=alpha, dropout=drop, target_modules=modules, merge=merge)


def w_datasets() -> DatasetConfig:
    sec("3 / 7  — Datasets")

    PRESETS = [
        ("OpceanAI/Yuuki-dataset",                "train","Yuuki dataset"),
        ("scryptiam/anime-waifu-personality-chat","train","Anime waifu personality"),
        ("TuringEnterprises/Open-RL",             "train","Open RL"),
    ]

    out = []
    print("  Preset datasets:")
    for ds_id, split, label in PRESETS:
        if ask_bool(f"  Include {label}", default=True):
            out.append(DatasetEntry(type="hf",id=ds_id,split=split,label=label))

    while ask_bool("\n  Add a HuggingFace dataset?", default=False):
        if HAS_HUB:
            q = input(f"  {B}>{RS} Search HF datasets (or Enter to type ID): ").strip()
            if q:
                try:
                    results = list(HfApi().list_datasets(search=q, limit=8))
                    if results:
                        print()
                        for i, r in enumerate(results,1):
                            print(f"    {C}{i}{RS}. {r.id}")
                        sel = input(f"  {B}>{RS} Pick # or type ID [{D}1{RS}]: ").strip()
                        ds_id = (results[int(sel)-1].id
                                 if sel.isdigit() and 1<=int(sel)<=len(results)
                                 else sel or results[0].id)
                    else:
                        warn("No results.")
                        ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                        if not ds_id: continue
                except Exception as e:
                    warn(f"Search error: {e}")
                    ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                    if not ds_id: continue
            else:
                ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                if not ds_id: continue
        else:
            ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
            if not ds_id: continue
        split = ask("Split", default="train")
        out.append(DatasetEntry(type="hf",id=ds_id,split=split,label=ds_id))
        ok(f"Added: {ds_id}")

    while ask_bool("\n  Add local file (JSON / JSONL / CSV)?", default=False):
        path = input(f"  {B}>{RS} File path: ").strip()
        if not os.path.isfile(path): err(f"Not found: {path}"); continue
        if Path(path).suffix.lower() not in (".json",".jsonl",".csv"):
            warn("Only .json .jsonl .csv supported"); continue
        out.append(DatasetEntry(type="local",path=path,label=Path(path).name))
        ok(f"Added: {path}")

    if not out: err("No datasets selected."); sys.exit(1)

    buf     = ask_int("Shuffle buffer size", default=1000, min_val=100)
    val_r   = ask_float("Validation split ratio  (0 = no eval)", default=0.05)
    preview = ask_bool("Preview 3 dataset examples before training", default=True)
    ext_tok = ask_bool("Extend tokenizer vocabulary from dataset", default=False)

    return DatasetConfig(
        datasets=out, buffer=buf,
        extend_tokenizer=ext_tok,
        preview_before_train=preview,
        val_split_ratio=max(0.0, min(0.5, val_r)),
    )


def w_hp() -> HyperParams:
    sec("4 / 7  — Hyperparameters")

    out_dir  = ask("Output directory", default="./yuuki_output")
    max_len  = ask_int("Max sequence length", default=512, min_val=64, max_val=8192)
    bs       = ask_int("Per-device batch size", default=1, min_val=1)
    ga       = ask_int("Gradient accumulation steps", default=8, min_val=1)
    ok(f"Effective batch = {bs*ga}")
    epochs   = ask_int("Epochs", default=3, min_val=1)
    ms       = ask_int("Max steps  (-1 = unlimited)", default=-1, min_val=-1)
    lr       = ask_float("Learning rate", default=2e-5)
    warmup   = ask_int("Warmup steps", default=100, min_val=0)
    wd       = ask_float("Weight decay", default=0.01)
    mgn      = ask_float("Max gradient norm", default=1.0)
    sc_c     = menu("LR scheduler", {
        "1":"cosine  ← recommended","2":"linear",
        "3":"constant","4":"cosine_with_restarts",
    }, default="1")
    sched    = {"1":"cosine","2":"linear","3":"constant","4":"cosine_with_restarts"}[sc_c]
    save_s   = ask_int("Save checkpoint every N steps", default=100, min_val=10)
    save_l   = ask_int("Max checkpoints to keep", default=3, min_val=1)
    eval_s   = ask_int("Eval perplexity every N steps  (0 = never)", default=200, min_val=0)
    seed     = ask_int("Random seed", default=42, min_val=0)
    smart_gc = ask_bool("Smart gradient checkpointing  (auto-enable if VRAM low)", default=True)
    adap_ck  = ask_bool("Adaptive checkpoint schedule  (more frequent at start)", default=True)

    return HyperParams(
        output_dir=out_dir, max_length=max_len, batch_size=bs, grad_accum=ga,
        epochs=epochs, max_steps=ms, lr=lr, warmup=warmup,
        weight_decay=wd, max_grad_norm=mgn, lr_scheduler=sched,
        save_steps=save_s, save_limit=save_l,
        eval_steps=eval_s if eval_s > 0 else 999999,
        seed=seed,
        smart_gradient_checkpointing=smart_gc,
        adaptive_checkpoint_schedule=adap_ck,
    )


def w_logging() -> LoggingConfig:
    sec("5a / 7  — Logging & Monitoring")
    use_wb = ask_bool("Log to Weights & Biases", default=False)
    wb_key, wb_proj = "", "yuuki-nxg"
    if use_wb:
        wb_proj = ask("W&B project name", default="yuuki-nxg")
        wb_key  = ask("W&B API key (Enter = use cached)", default="")
    use_tb = ask_bool("Log to TensorBoard", default=False)
    tb_dir = "./runs"
    if use_tb:
        tb_dir = ask("TensorBoard log dir", default="./runs")
    return LoggingConfig(
        use_wandb=use_wb, wandb_project=wb_proj, wandb_api_key=wb_key,
        use_tensorboard=use_tb, tensorboard_dir=tb_dir,
    )


def w_webhook() -> WebhookConfig:
    sec("5b / 7  — Webhook Notifications")
    if not ask_bool("Enable webhook notifications", default=False):
        return WebhookConfig()
    platform = menu("Platform", {
        "1":"Discord","2":"Slack","3":"Telegram","4":"ntfy.sh","5":"Custom"
    })
    platform = {"1":"discord","2":"slack","3":"telegram","4":"ntfy","5":"custom"}[platform]
    url      = ask("Webhook URL")
    on_fin   = ask_bool("Notify on training complete", default=True)
    on_cr    = ask_bool("Notify on crash", default=True)
    on_ck    = ask_bool("Notify on each checkpoint", default=False)
    return WebhookConfig(enabled=True, url=url, platform=platform,
                         on_finish=on_fin, on_crash=on_cr, on_checkpoint=on_ck)


def w_dpo() -> DPOConfig:
    sec("5c / 7  — DPO / Alignment  (optional)")
    if not ask_bool("Enable DPO training  (Direct Preference Optimization)", default=False):
        return DPOConfig()
    info("DPO needs a dataset with 'prompt', 'chosen', 'rejected' columns.")
    ds_id      = ask("DPO dataset ID", default="Anthropic/hh-rlhf")
    beta       = ask_float("Beta  (higher = closer to base model)", default=0.1)
    max_prompt = ask_int("Max prompt length", default=256, min_val=32)
    max_len    = ask_int("Max total length", default=512, min_val=64)
    return DPOConfig(enabled=True, dataset_id=ds_id, beta=beta,
                     max_prompt_length=max_prompt, max_length=max_len)


def w_optuna() -> OptimizationConfig:
    if not ask_bool("Run Optuna hyperparameter search before full training?", default=False):
        return OptimizationConfig()
    n = ask_int("Number of trials  (more = better but slower)", default=10, min_val=2, max_val=100)
    search_lr  = ask_bool("Search learning rate?", default=True)
    search_bs  = ask_bool("Search batch size?", default=False)
    search_r   = ask_bool("Search LoRA rank?", default=False)
    return OptimizationConfig(
        use_optuna=True, optuna_trials=n,
        search_lr=search_lr, search_batch=search_bs, search_lora_r=search_r,
    )


def w_post() -> PostTrainingConfig:
    sec("6 / 7  — Post-Training Actions")

    chat    = ask_bool("Interactive chat test after training", default=True)
    gguf    = False; gguf_q = "Q4_K_M"; gguf_t = "ollama"
    onnx    = False
    gptq    = False; gptq_b = 4
    upload  = False; hf_repo = ""; hf_tok = ""; hf_priv = False
    synth   = False; synth_n = 1000

    if ask_bool("Convert to GGUF  (Ollama / llama.cpp)", default=False):
        gguf  = True
        qf    = menu("GGUF quantization", {
            "1":"Q4_K_M  ← recommended","2":"Q5_K_M  — better quality",
            "3":"Q8_0    — near-lossless","4":"f16     — full precision",
        }, default="1")
        gguf_q = {"1":"Q4_K_M","2":"Q5_K_M","3":"Q8_0","4":"f16"}[qf]
        tgt    = menu("Target runtime", {"1":"Ollama","2":"llama.cpp","3":"Both"}, default="1")
        gguf_t = {"1":"ollama","2":"llamacpp","3":"both"}[tgt]

    if ask_bool("Export to ONNX  (fast CPU inference)", default=False):
        onnx = True

    if ask_bool("Convert to GPTQ  (GPU quantized deployment)", default=False):
        gptq   = True
        gptq_b = ask_int("GPTQ bits", default=4, min_val=2, max_val=8)

    if ask_bool("Generate synthetic training data first", default=False):
        synth   = True
        synth_n = ask_int("Number of synthetic samples", default=1000, min_val=10)

    if ask_bool("Upload to HuggingFace Hub", default=False):
        if not HAS_HUB:
            warn("huggingface_hub not installed — skipping.")
        else:
            upload   = True
            hf_repo  = ask("Destination repo  (e.g. OpceanAI/Yuuki-finetuned)")
            hf_tok   = ask("HF write token  (Enter = cached login)", default="")
            hf_priv  = ask_bool("Private repo", default=False)

    return PostTrainingConfig(
        chat_test=chat,
        convert_gguf=gguf, gguf_quant=gguf_q, gguf_target=gguf_t,
        convert_onnx=onnx,
        convert_gptq=gptq, gptq_bits=gptq_b,
        upload_hf=upload, hf_repo=hf_repo,
        hf_token=hf_tok, hf_private=hf_priv,
        synthetic_data=synth, synthetic_samples=synth_n,
    )


def w_summary(cfg: TrainingConfig):
    sec("7 / 7  — Summary & Cost Estimate")

    for k, v in cfg.summary_dict().items():
        print(f"  {B}{k:<22}{RS} {v}")

    print(f"\n  {B}{Y}{'─'*62}{RS}")
    print(f"  {B}{Y}  Training Time Estimate{RS}")
    print(f"  {B}{Y}{'─'*62}{RS}")

    try:
        from core.estimator import build_estimate_report
        print(build_estimate_report(cfg))
    except Exception as e:
        warn(f"Estimate unavailable: {e}")

    errors = cfg.validate()
    if errors:
        print()
        for e in errors: err(e)
        sys.exit(1)

    print()
    if not ask_bool("Start training", default=True):
        print("  Cancelled."); sys.exit(0)


def run_training(cfg: TrainingConfig):
    import threading
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from transformers.trainer_utils import get_last_checkpoint
    from core.model_utils import load_model_and_tokenizer, generate_synthetic_data
    from core.dataset_utils import build_dataset, preview_dataset, extend_tokenizer_vocabulary
    from core.callbacks import build_callbacks, WebhookNotifier
    from core.post_training import run_post_training
    from core.dpo_utils import run_dpo_training, run_optuna_search, apply_optuna_results

    out_dir = cfg.hp.output_dir
    os.makedirs(out_dir, exist_ok=True)

    cfg.save()
    ok(f"Config saved to {out_dir}/training_config.json")

    stop_flag = [False]
    webhook   = next((c for c in [] if isinstance(c, WebhookNotifier)), None)

    def _sig(sig, frame):
        stop_flag[0] = True
        logger.warning("SIGINT — requesting stop after current step...")
    signal.signal(signal.SIGINT, _sig)

    last_ckpt = get_last_checkpoint(out_dir)
    if last_ckpt: ok(f"Resuming from: {last_ckpt}")

    if cfg.post.synthetic_data:
        sec("Pre-Training — Synthetic Data Generation")
        synth = generate_synthetic_data(cfg.model.model_path, cfg.post.synthetic_samples)
        if synth:
            import json
            synth_path = os.path.join(out_dir, "synthetic_data.jsonl")
            with open(synth_path, "w") as f:
                for s in synth: f.write(json.dumps(s) + "\n")
            ok(f"Saved {len(synth)} synthetic samples: {synth_path}")
            from core.config import DatasetEntry
            cfg.dataset.datasets.append(DatasetEntry(type="local", path=synth_path, label="synthetic"))

    model, tokenizer, device = load_model_and_tokenizer(cfg.model, cfg.hw)

    if cfg.dataset.extend_tokenizer:
        sec("Tokenizer Vocabulary Extension")
        try:
            from datasets import load_dataset as _ld
            sample_ds = _ld(cfg.dataset.datasets[0].id, split="train",
                            streaming=True, trust_remote_code=True)
            tokenizer, added = extend_tokenizer_vocabulary(tokenizer, sample_ds)
            if added > 0:
                model.resize_token_embeddings(len(tokenizer))
                ok(f"Vocabulary extended by {added} tokens.")
        except Exception as e:
            warn(f"Vocabulary extension failed: {e}")

    train_ds, fmt_type = build_dataset(cfg.dataset, tokenizer, cfg.hp.max_length,
                                        cfg.dataset.val_split_ratio)
    ok(f"Dataset format detected: {fmt_type}")

    if cfg.dataset.preview_before_train:
        sec("Dataset Preview")
        try:
            previews = preview_dataset(train_ds, n=3)
            for i, p in enumerate(previews, 1):
                print(f"\n  {B}{C}─── Example {i} ───{RS}")
                print(f"  {p[:500]}")
            print()
            if not ask_bool("Looks good? Continue with training", default=True):
                print("  Cancelled."); sys.exit(0)
        except Exception as e:
            warn(f"Preview failed: {e}")

    if cfg.optim.use_optuna:
        sec("Optuna Hyperparameter Search")

        def _build_model(trial_cfg):
            return load_model_and_tokenizer(trial_cfg.model, trial_cfg.hw)

        def _build_data(trial_cfg, tok):
            ds, _ = build_dataset(trial_cfg.dataset, tok, trial_cfg.hp.max_length)
            return ds

        best = run_optuna_search(_build_model, _build_data, cfg, cfg.optim.optuna_trials)
        if best:
            cfg = apply_optuna_results(cfg, best)
            ok(f"Applied best hyperparameters: {best}")
            model, tokenizer, device = load_model_and_tokenizer(cfg.model, cfg.hw)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    mode, q = cfg.hw.mode, cfg.hw.quantization
    if device=="cuda" and HAS_BNB and q=="none" and mode=="full":   optim="adamw_8bit"
    elif device=="cuda" and mode in ("lora","qlora"):               optim="paged_adamw_8bit" if HAS_BNB else "adamw_torch"
    elif device=="cuda":                                            optim="adamw_torch_fused"
    else:                                                           optim="adamw_torch"

    report_to = []
    if cfg.logging.use_wandb:
        if cfg.logging.wandb_api_key:
            os.environ["WANDB_API_KEY"] = cfg.logging.wandb_api_key
        os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
        report_to.append("wandb")
    if cfg.logging.use_tensorboard:
        report_to.append("tensorboard")

    t_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=False,
        num_train_epochs=cfg.hp.epochs,
        max_steps=cfg.hp.max_steps if cfg.hp.max_steps > 0 else -1,
        per_device_train_batch_size=cfg.hp.batch_size,
        gradient_accumulation_steps=cfg.hp.grad_accum,
        learning_rate=cfg.hp.lr,
        warmup_steps=cfg.hp.warmup,
        weight_decay=cfg.hp.weight_decay,
        lr_scheduler_type=cfg.hp.lr_scheduler,
        fp16=(device=="cuda" and cfg.hw.precision=="fp16"),
        bf16=(device=="cuda" and cfg.hw.precision=="bf16"),
        tf32=(device=="cuda"),
        save_strategy="steps",
        save_steps=cfg.hp.save_steps,
        save_total_limit=cfg.hp.save_limit,
        eval_strategy="steps" if cfg.dataset.val_split_ratio > 0 else "no",
        eval_steps=cfg.hp.eval_steps,
        logging_steps=10,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device=="cuda"),
        remove_unused_columns=False,
        report_to=report_to if report_to else [],
        gradient_checkpointing=(q=="none" and mode=="full"),
        optim=optim,
        max_grad_norm=cfg.hp.max_grad_norm,
        seed=cfg.hp.seed,
        logging_dir=cfg.logging.tensorboard_dir if cfg.logging.use_tensorboard else None,
    )

    total_steps = t_args.max_steps if t_args.max_steps and t_args.max_steps > 0 else None
    callbacks   = build_callbacks(cfg, out_dir, total_steps, stop_flag)

    trainer = Trainer(
        model=model,
        args=t_args,
        data_collator=collator,
        train_dataset=train_ds,
        callbacks=callbacks,
    )

    wb_notifier = next((c for c in callbacks if hasattr(c, "notify_crash")), None)

    if cfg.dpo.enabled:
        run_dpo_training(model, tokenizer, cfg, out_dir, stop_flag)
    else:
        try:
            trainer.train(resume_from_checkpoint=last_ckpt)
        except KeyboardInterrupt:
            logger.warning("Interrupted — saving emergency checkpoint...")
            step = getattr(trainer.state, "global_step", 0)
            dest = os.path.join(out_dir, f"emergency-step{step}")
            trainer.save_model(dest); tokenizer.save_pretrained(dest)
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Training crashed: {e}")
            if wb_notifier: wb_notifier.notify_crash(str(e))
            step = getattr(trainer.state, "global_step", 0)
            dest = os.path.join(out_dir, f"crash-step{step}")
            try: trainer.save_model(dest); tokenizer.save_pretrained(dest)
            except: pass
            raise

    final_dir = os.path.join(out_dir, "final")
    ok(f"Saving final model → {final_dir}")

    if mode in ("lora","qlora") and cfg.lora.merge:
        try:
            model.merge_and_unload().save_pretrained(final_dir)
            ok("LoRA weights merged into base model.")
        except Exception as e:
            warn(f"Merge failed: {e}"); trainer.save_model(final_dir)
    else:
        trainer.save_model(final_dir)

    tokenizer.save_pretrained(final_dir)
    ok("Training complete! 🎉")

    run_post_training(final_dir, cfg, tokenizer, wb_notifier)


def main():
    banner()

    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        if os.path.isfile(cfg_path):
            info(f"Loading config from: {cfg_path}")
            cfg = TrainingConfig.load(cfg_path)
            w_summary(cfg)
            run_training(cfg)
            return

    model_cfg    = w_model()
    hw_cfg       = w_hw(is_gguf=not model_cfg.model_is_hf)
    lora_cfg     = w_lora(hw_cfg.mode)
    hw_cfg_dict  = hw_cfg.__dict__.copy()
    hw_cfg_dict["lora_cfg"] = lora_cfg
    ds_cfg       = w_datasets()
    hp_cfg       = w_hp()
    log_cfg      = w_logging()
    wh_cfg       = w_webhook()
    dpo_cfg      = w_dpo()
    optuna_cfg   = w_optuna()
    post_cfg     = w_post()

    cfg = TrainingConfig(
        model=model_cfg,
        hw=hw_cfg,
        lora=lora_cfg,
        dataset=ds_cfg,
        hp=hp_cfg,
        logging=log_cfg,
        webhook=wh_cfg,
        dpo=dpo_cfg,
        optim=optuna_cfg,
        post=post_cfg,
    )

    w_summary(cfg)

    if not model_cfg.model_is_hf and model_cfg.gguf_path:
        sec("GGUF Inference Setup")
        from core.post_training import convert_to_gguf
        gguf_dir = os.path.join(hp_cfg.output_dir, "gguf")
        os.makedirs(gguf_dir, exist_ok=True)
        from core.post_training import _generate_modelfile
        short = Path(model_cfg.gguf_path).stem
        _generate_modelfile(gguf_dir, model_cfg.gguf_path, short)
        return

    run_training(cfg)


if __name__ == "__main__":
    main()
