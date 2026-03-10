#!/usr/bin/env python3

import os, sys, json, csv, signal, logging, threading, shutil, time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import bitsandbytes
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
try:
    from huggingface_hub import HfApi, hf_hub_download, login as hf_login
    HAS_HUB = True
except ImportError:
    HAS_HUB = False

from datasets import load_dataset, interleave_datasets, Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_yuuki")

_trainer_ref   = None
_tokenizer_ref = None
_save_lock     = threading.Lock()
_stop_training = False

C  = "\033[96m"
G  = "\033[92m"
Y  = "\033[93m"
R  = "\033[91m"
B  = "\033[1m"
D  = "\033[2m"
RS = "\033[0m"

def banner():
    print(f"\n{B}{C}  ██╗   ██╗██╗   ██╗██╗   ██╗██╗  ██╗██╗")
    print(f"  \u255a\u2550\u2550\u2557 \u2554\u255d\u2550\u2550\u2550\u2550\u2550\u2550\u2557\u2551   \u2551\u2551\u2551   \u2551\u2551\u2551 \u2554\u255d\u2551")
    print(f"   \u255a\u2550\u2550\u2550\u2554\u255d \u2551   \u2551\u2551\u2551   \u2551\u2551\u2551\u2550\u2550\u2550\u2554\u255d \u2551")
    print(f"    \u255a\u2550\u2554\u255d  \u2551   \u2551\u2551\u2551   \u2551\u2551\u2551\u2554\u2550\u2557\u2551 \u2551")
    print(f"     \u2551\u2551   \u255a\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u255d \u2551\u2551  \u2551\u2551\u2551")
    print(f"     \u255a\u255d    \u255a\u2550\u2550\u2550\u2550\u255d  \u255a\u2550\u2550\u2550\u2550\u255d \u255a\u255d  \u255a\u255d\u255a\u255d")
    print(f"       NxG Training Wizard  \u2014  OpceanAI{RS}\n")

def sec(title):
    print(f"\n{B}{Y}{'─'*60}{RS}\n{B}{Y}  {title}{RS}\n{B}{Y}{'─'*60}{RS}\n")

def ok(m):   print(f"  {G}\u2713{RS} {m}")
def warn(m): print(f"  {Y}\u26a0{RS} {m}")
def err(m):  print(f"  {R}\u2717{RS} {m}")

def _hs(b):
    if b < 1024:   return f"{b} B"
    if b < 1<<20:  return f"{b/1024:.1f} KB"
    if b < 1<<30:  return f"{b/(1<<20):.1f} MB"
    return f"{b/(1<<30):.2f} GB"

def dbar(desc, cur, total, width=38):
    frac   = min(cur/total, 1.0) if total > 0 else 0
    filled = int(width * frac)
    bar    = ":" * filled + " " * (width - filled)
    pct    = f"{frac*100:5.1f}%" if total > 0 else "  ..."
    print(f"\r  {C}{desc:<28}{RS} [{G}{bar}{RS}] {pct}  {_hs(cur)}/{_hs(total) if total>0 else '?'}", end="", flush=True)

def ddone(desc):
    print(f"\r  {C}{desc:<28}{RS} [{G}{'='*38}{RS}] {G}Done \u2713{RS}        ")

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
        arrow = f"{G}\u2192{RS}" if k == default else " "
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
            info = api.get_paths_info(repo_id, [fname])
            size = info[0].size if info else 0
        except: size = 0
        short = (fname[:26]+"..") if len(fname) > 28 else fname
        dbar(short, 0, size)
        hf_hub_download(repo_id=repo_id, filename=fname, local_dir=local_dir)
        ddone(short)
    return local_dir

def hf_dl_gguf(repo_id, filename, dest_dir):
    if not HAS_HUB: raise RuntimeError("huggingface_hub not installed.")
    os.makedirs(dest_dir, exist_ok=True)
    short = (filename[:26]+"..") if len(filename) > 28 else filename
    dbar(short, 0, 0)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
    ddone(short)
    return path


def w_model():
    sec("1 / 6  — Model Source")
    c = menu("Where is your model?", {
        "1": "HuggingFace ID  (auto-download on first use)",
        "2": "Download from HuggingFace now  (with progress bar)",
        "3": "Local directory  (already downloaded)",
        "4": "Local / HF  .gguf  (inference + Ollama/llama.cpp setup)",
        "5": "Local  .pt / .safetensors / .bin  file",
    })
    model_path, gguf_path, is_hf = None, None, True

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
        warn("GGUF cannot be fine-tuned with transformers — inference/setup only.")
    elif c == "5":
        model_path = ask("Path to file or directory")
        if not os.path.exists(model_path): err(f"Not found: {model_path}"); sys.exit(1)

    return {"model_path": model_path, "gguf_path": gguf_path, "model_is_hf": is_hf}


def w_hw(is_gguf=False):
    sec("2 / 6  — Training Mode & Hardware")
    if is_gguf:
        ok("GGUF selected — skipping training config.")
        return {"mode":"none","use_gpu":False,"precision":"fp32","quantization":"none","flash_attn":False,"compile":False}

    cuda_ok = HAS_TORCH and torch.cuda.is_available()
    mps_ok  = HAS_TORCH and hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory/1e9,1)
        ok(f"CUDA: {name}  ({vram} GB VRAM)")
    elif mps_ok: ok("Apple MPS GPU detected.")
    else: warn("No GPU — CPU only.")

    mode_c = menu("Training mode", {
        "1": "Full fine-tune   — all weights, maximum VRAM",
        "2": "LoRA             — adapters only, fast & low RAM",
        "3": "QLoRA            — LoRA + 4-bit base, minimum VRAM",
    }, default="2")
    mode = {"1":"full","2":"lora","3":"qlora"}[mode_c]

    use_gpu, precision, quant, flash_attn, compile_m = False, "fp32", "none", False, False

    if cuda_ok:  use_gpu = ask_bool("Use GPU (CUDA)", default=True)
    if use_gpu and cuda_ok:
        if mode == "qlora":
            precision, quant = "bf16", "int4"
            ok("QLoRA: auto-set bf16 + int4 NF4.")
        else:
            pc = menu("Precision", {
                "1":"fp16  — fast, standard",
                "2":"bf16  — stable (Ampere+)",
                "3":"fp32  — full precision, slow",
            })
            precision = {"1":"fp16","2":"bf16","3":"fp32"}[pc]
            if mode == "full" and HAS_BNB:
                qc = menu("Weight quantization (base model only)", {
                    "1":"none  — full weights",
                    "2":"int8  — 8-bit bitsandbytes",
                }, default="1")
                quant = {"1":"none","2":"int8"}[qc]

        try:
            import flash_attn
            flash_attn = ask_bool("Flash Attention 2", default=True)
        except ImportError: warn("flash-attn not installed.")

        compile_m = ask_bool("torch.compile (PyTorch 2.0+)", default=False)

    elif mps_ok:
        use_gpu, precision = True, "fp32"
        warn("MPS: using fp32.")

    return {"mode":mode,"use_gpu":use_gpu,"precision":precision,
            "quantization":quant,"flash_attn":flash_attn,"compile":compile_m}


def w_lora(mode):
    if mode not in ("lora","qlora"): return {}
    sec("2b / 6  — LoRA / QLoRA Config")
    r      = ask_int("Rank (r)", default=16, min_val=1)
    alpha  = ask_int("Alpha", default=r*2, min_val=1)
    drop   = ask_float("Dropout", default=0.05)
    print(f"\n  Target modules  {D}(Enter = auto  |  e.g. q_proj,v_proj,k_proj,o_proj){RS}")
    raw = input(f"  {B}>{RS} Modules: ").strip()
    modules = [m.strip() for m in raw.split(",")] if raw else None
    merge = ask_bool("Merge adapters into base model after training", default=False) if mode=="lora" else False
    return {"r":r,"alpha":alpha,"dropout":drop,"target_modules":modules,"merge":merge}


def w_datasets():
    sec("3 / 6  — Datasets")
    PRESETS = [
        ("OpceanAI/Yuuki-dataset",               "train","Yuuki dataset"),
        ("scryptiam/anime-waifu-personality-chat","train","Anime waifu personality"),
        ("TuringEnterprises/Open-RL",            "train","Open RL"),
    ]
    out = []
    print("  Preset datasets:")
    for ds_id, split, label in PRESETS:
        if ask_bool(f"  Include {label}", default=True):
            out.append({"type":"hf","id":ds_id,"config":None,"label":label,"split":split})

    while ask_bool("\n  Add a HuggingFace dataset?", default=False):
        if HAS_HUB:
            q = input(f"  {B}>{RS} Search HF (or Enter to type ID): ").strip()
            if q:
                try:
                    results = list(HfApi().list_datasets(search=q, limit=8))
                    if results:
                        print()
                        for i, r in enumerate(results, 1):
                            print(f"    {C}{i}{RS}. {r.id}")
                        sel = input(f"  {B}>{RS} Pick number or type ID [{D}1{RS}]: ").strip()
                        ds_id = results[int(sel)-1].id if sel.isdigit() and 1<=int(sel)<=len(results) else (sel or results[0].id)
                    else:
                        warn("No results."); ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                        if not ds_id: continue
                except Exception as e:
                    warn(f"Search error: {e}"); ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                    if not ds_id: continue
            else:
                ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
                if not ds_id: continue
        else:
            ds_id = input(f"  {B}>{RS} Dataset ID: ").strip()
            if not ds_id: continue
        split = ask("Split", default="train")
        out.append({"type":"hf","id":ds_id,"config":None,"label":ds_id,"split":split})
        ok(f"Added: {ds_id}")

    while ask_bool("\n  Add local file (JSON / JSONL / CSV)?", default=False):
        path = input(f"  {B}>{RS} File path: ").strip()
        if not os.path.isfile(path): err(f"Not found: {path}"); continue
        if Path(path).suffix.lower() not in (".json",".jsonl",".csv"):
            warn("Only .json .jsonl .csv supported"); continue
        out.append({"type":"local","path":path,"label":Path(path).name})
        ok(f"Added: {path}")

    if not out: err("No datasets selected."); sys.exit(1)
    buf = ask_int("Streaming shuffle buffer size", default=1000, min_val=100)
    return {"datasets":out,"buffer":buf}


def w_hp():
    sec("4 / 6  — Hyperparameters")
    out_dir   = ask("Output directory", default="./yuuki_nxg_output")
    max_len   = ask_int("Max sequence length", default=512, min_val=64, max_val=8192)
    bs        = ask_int("Per-device batch size", default=1, min_val=1)
    ga        = ask_int("Gradient accumulation steps", default=8, min_val=1)
    ok(f"Effective batch = {bs * ga}")
    epochs    = ask_int("Epochs", default=3, min_val=1)
    max_steps = ask_int("Max steps  (-1 = unlimited)", default=-1, min_val=-1)
    lr        = ask_float("Learning rate", default=2e-5)
    warmup    = ask_int("Warmup steps", default=100, min_val=0)
    wd        = ask_float("Weight decay", default=0.01)
    mgn       = ask_float("Max gradient norm", default=1.0)
    sc_c      = menu("LR scheduler", {
        "1":"cosine","2":"linear","3":"constant","4":"cosine_with_restarts"
    }, default="1")
    sched     = {"1":"cosine","2":"linear","3":"constant","4":"cosine_with_restarts"}[sc_c]
    save_s    = ask_int("Save every N steps", default=100, min_val=10)
    save_l    = ask_int("Max checkpoints to keep", default=3, min_val=1)
    seed      = ask_int("Random seed", default=42, min_val=0)
    return {
        "output_dir":out_dir,"max_length":max_len,"batch_size":bs,"grad_accum":ga,
        "epochs":epochs,"max_steps":max_steps,"lr":lr,"warmup":warmup,
        "weight_decay":wd,"max_grad_norm":mgn,"lr_scheduler":sched,
        "save_steps":save_s,"save_limit":save_l,"seed":seed,
    }


def w_post():
    sec("5 / 6  — Post-Training Actions")
    actions = []

    if ask_bool("Convert to GGUF + prepare llama.cpp / Ollama", default=False):
        qf  = menu("GGUF quantization", {
            "1":"Q4_K_M  — recommended","2":"Q5_K_M  — better quality",
            "3":"Q8_0    — near-lossless","4":"f16     — full precision GGUF",
        }, default="1")
        tgt = menu("Runtime target", {"1":"Ollama","2":"llama.cpp","3":"Both"}, default="1")
        actions.append({
            "action":"gguf",
            "quant": {"1":"Q4_K_M","2":"Q5_K_M","3":"Q8_0","4":"f16"}[qf],
            "target":{"1":"ollama","2":"llamacpp","3":"both"}[tgt],
        })

    if ask_bool("Upload final model to HuggingFace Hub", default=False):
        if not HAS_HUB: warn("huggingface_hub not installed — skipping.")
        else:
            repo  = ask("Destination repo (e.g. YourOrg/Yuuki-finetuned)")
            token = ask("HF write token (Enter = cached login)", default="")
            priv  = ask_bool("Private repo", default=False)
            actions.append({"action":"hf_upload","repo":repo,"token":token or None,"private":priv})

    return actions


def w_summary(cfg):
    sec("6 / 6  — Summary")
    m, hw, hp, ds = cfg["model"], cfg["hw"], cfg["hp"], cfg["ds"]
    lc = cfg.get("lora_cfg",{})
    print(f"  {'Model':<22} {m['model_path'] or m['gguf_path']}")
    print(f"  {'Mode':<22} {hw['mode'].upper()}")
    if lc: print(f"  {'LoRA r / alpha':<22} {lc.get('r')} / {lc.get('alpha')}")
    gpu_s = f"GPU  {hw['precision']} / quant={hw['quantization']}" if hw['use_gpu'] else "CPU"
    print(f"  {'Device':<22} {gpu_s}")
    for k, v in [("Flash Attention","flash_attn"),("torch.compile","compile")]:
        print(f"  {k:<22} {'yes' if hw.get(v) else 'no'}")
    print(f"  {'Output dir':<22} {hp['output_dir']}")
    print(f"  {'Seq length':<22} {hp['max_length']}")
    print(f"  {'Effective batch':<22} {hp['batch_size']}x{hp['grad_accum']} = {hp['batch_size']*hp['grad_accum']}")
    print(f"  {'Epochs':<22} {hp['epochs']}")
    print(f"  {'Max steps':<22} {'unlimited' if hp['max_steps']==-1 else hp['max_steps']}")
    print(f"  {'LR / scheduler':<22} {hp['lr']}  /  {hp['lr_scheduler']}")
    print(f"  {'Save every':<22} {hp['save_steps']} steps")
    print(f"  {'Datasets':<22} {', '.join(d['label'] for d in ds['datasets'])}")
    for pa in cfg.get("post_actions",[]):
        if pa["action"]=="gguf": print(f"  {'Post GGUF':<22} {pa['quant']}  ->  {pa['target']}")
        elif pa["action"]=="hf_upload": print(f"  {'Post Upload':<22} {pa['repo']}")
    print()
    if not ask_bool("Start training", default=True):
        print("  Cancelled."); sys.exit(0)


def detect_col(features):
    for col in ["text","content","code","conversation","prompt","input","instruction","response","message","question","answer"]:
        if col in features: return col
    for col in features:
        if "string" in str(getattr(features[col],"dtype","")).lower(): return col
    return list(features.keys())[0]

def flatten(val):
    if isinstance(val, str): return val.strip()
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                r = item.get("role", item.get("from",""))
                c = item.get("content", item.get("value", item.get("text","")))
                parts.append(f"{r}: {c}" if r and c else str(c or ""))
            elif isinstance(item, str): parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(val, dict):
        return "\n".join(f"{k}: {v}" for k,v in val.items() if isinstance(v,str)).strip()
    return str(val).strip() if val else ""

def load_local(path):
    ext = Path(path).suffix.lower()
    ok(f"Loading: {path}")
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return HFDataset.from_list(list(csv.DictReader(f)))
    rows = []
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except: pass
    return HFDataset.from_list(rows)

def build_data(ds_cfg, tokenizer, max_len):
    streams, statics = [], []
    for ds in ds_cfg["datasets"]:
        try:
            if ds["type"] == "hf":
                logger.info(f"Loading {ds['label']}...")
                d = load_dataset(ds["id"], ds.get("config"), split=ds.get("split","train"),
                                 streaming=True, trust_remote_code=True)
                streams.append(d)
            else:
                statics.append(load_local(ds["path"]))
        except Exception as e:
            warn(f"Could not load {ds.get('label','?')}: {e}")

    if not streams and not statics: raise RuntimeError("No datasets loaded.")

    cs = None
    if streams:
        cs = interleave_datasets(streams, stopping_strategy="all_exhausted") if len(streams)>1 else streams[0]
        cs = cs.shuffle(buffer_size=ds_cfg["buffer"], seed=42)

    sample = next(iter(cs if cs else statics[0]))
    text_col = detect_col(sample)
    all_cols = list(sample.keys())
    logger.info(f"Text column: '{text_col}'")

    def tok(example):
        text = flatten(example.get(text_col,""))
        if not text: return {"input_ids":[],"attention_mask":[],"labels":[]}
        out = tokenizer(text, truncation=True, max_length=max_len, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    parts = []
    if cs:
        t = cs.map(tok, remove_columns=all_cols)
        parts.append(t.filter(lambda x: len(x["input_ids"])>0))
    for sd in statics:
        t = sd.map(tok, remove_columns=sd.column_names)
        parts.append(t.filter(lambda x: len(x["input_ids"])>0))

    if len(parts) == 1: return parts[0]
    return interleave_datasets(parts, stopping_strategy="all_exhausted")


def load_model(model_cfg, hw_cfg):
    mp = model_cfg["model_path"]
    logger.info(f"Loading tokenizer: {mp}")
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    kw = {"trust_remote_code": True}
    device = "cuda" if hw_cfg["use_gpu"] and HAS_TORCH and torch.cuda.is_available() else "cpu"

    if device == "cuda":
        kw["device_map"] = "auto"
        q, prec = hw_cfg["quantization"], hw_cfg["precision"]
        if q == "int4" or hw_cfg["mode"] == "qlora":
            cdtype = torch.bfloat16 if prec=="bf16" else torch.float16
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=cdtype,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
        elif q == "int8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kw["torch_dtype"] = torch.bfloat16 if prec=="bf16" else torch.float16 if prec=="fp16" else torch.float32
        if hw_cfg.get("flash_attn"):
            kw["attn_implementation"] = "flash_attention_2"

    logger.info(f"Loading model: {mp}")
    model = AutoModelForCausalLM.from_pretrained(mp, **kw)

    if hw_cfg["mode"] in ("lora","qlora"):
        try:
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
            if hw_cfg["mode"] == "qlora":
                model = prepare_model_for_kbit_training(model)
            lc = hw_cfg.get("lora_cfg",{})
            tgt = lc.get("target_modules") or ["q_proj","v_proj","k_proj","o_proj"]
            model = get_peft_model(model, LoraConfig(
                r=lc.get("r",16), lora_alpha=lc.get("alpha",32),
                target_modules=tgt, lora_dropout=lc.get("dropout",0.05),
                bias="none", task_type=TaskType.CAUSAL_LM,
            ))
            model.print_trainable_parameters()
        except ImportError:
            warn("peft not installed — falling back to full fine-tune.")

    if hw_cfg.get("compile") and device=="cuda":
        try: model = torch.compile(model); ok("Model compiled.")
        except Exception as e: warn(f"torch.compile failed: {e}")

    if hw_cfg["quantization"]=="none" and hw_cfg["mode"]=="full":
        try: model.gradient_checkpointing_enable()
        except: pass

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"Params: {total:,} total  |  {trainable:,} trainable ({100*trainable/total:.2f}%)")
    return model, tok, device


def emerg(output_dir, tag="emergency"):
    with _save_lock:
        if _trainer_ref is None: return
        try:
            step = getattr(_trainer_ref.state,"global_step",0)
            dest = os.path.join(output_dir, f"manual-{tag}-step{step}")
            logger.info(f"Emergency save → {dest}")
            _trainer_ref.save_model(dest)
            if _tokenizer_ref: _tokenizer_ref.save_pretrained(dest)
            ok("Saved.")
        except Exception as e: err(f"Save failed: {e}")


class ResMonCB(TrainerCallback):
    def __init__(self, every=50, mem=0.12, cpu=95):
        self.every=every; self.mem=mem; self.cpu=cpu
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.every != 0 or psutil is None: return control
        try:
            vm = psutil.virtual_memory(); av = vm.available/vm.total
            cp = psutil.cpu_percent(interval=None)
            if av < self.mem or cp >= self.cpu:
                logger.warning(f"Step {state.global_step}: mem={av:.2f} cpu={cp:.0f}% -> save")
                control.should_save = True
        except: pass
        return control

class DockerCB(TrainerCallback):
    def __init__(self): self._t0=None; self._s0=None; self._total=None
    def on_train_begin(self, args, state, control, **kw):
        self._total = state.max_steps if state.max_steps and state.max_steps>0 else None
        self._s0 = state.global_step; self._t0 = time.time(); print()
    def on_step_end(self, args, state, control, **kw):
        step = state.global_step; total = self._total or 0
        loss = next((e["loss"] for e in reversed(state.log_history) if "loss" in e), None)
        elapsed = time.time() - (self._t0 or time.time())
        done = step - (self._s0 or 0)
        eta = ""
        if done > 0 and total > 0:
            s = elapsed/done*(total-step); h,r=divmod(int(s),3600); m,sc=divmod(r,60)
            eta = f"  ETA {h:02d}:{m:02d}:{sc:02d}"
        W = 36
        if total > 0:
            fr=step/total; fi=int(W*fr); bar=":"*fi+" "*(W-fi); pct=f"{fr*100:5.1f}%"
        else:
            fi=step%W; bar=" "*fi+":::"+" "*max(0,W-fi-3); pct=f"step {step}"
        ls = f"  loss {loss:.4f}" if loss else ""
        print(f"\r  {C}Training{RS} [{G}{bar}{RS}] {pct}{ls}{eta}   ", end="", flush=True)
    def on_train_end(self, args, state, control, **kw):
        print(f"\r  {C}Training{RS} [{G}{'='*36}{RS}] {G}Done \u2713{RS}                              ")

class StopCB(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if _stop_training: control.should_training_stop = True
        return control

class CkptCB(TrainerCallback):
    def __init__(self, d, limit): self.d=d; self.limit=limit
    def on_save(self, args, state, control, **kw):
        if not os.path.exists(self.d): return control
        ckpts = sorted([
            (int(d.split("-")[-1]), os.path.join(self.d, d))
            for d in os.listdir(self.d)
            if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
        ], reverse=True)
        for _, p in ckpts[self.limit:]:
            try: shutil.rmtree(p); logger.info(f"Removed: {p}")
            except Exception as e: warn(f"Could not remove {p}: {e}")
        return control


def do_gguf(final_dir, model_path, action):
    sec("Post-Training  — GGUF Conversion")
    script = next((p for p in [
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "/usr/local/lib/python3.11/dist-packages/llama_cpp/convert_hf_to_gguf.py",
    ] if os.path.isfile(p)), None)

    gguf_dir = os.path.join(final_dir,"gguf"); os.makedirs(gguf_dir, exist_ok=True)
    gguf_out = os.path.join(gguf_dir, f"model-{action['quant']}.gguf")
    short    = Path(model_path).name if model_path else "yuuki"

    if script:
        ok(f"Converting to GGUF {action['quant']}...")
        ret = os.system(f"python {script} {final_dir} --outfile {gguf_out} --outtype {action['quant'].lower()}")
        if ret != 0: warn("Conversion failed."); return
        ok(f"GGUF saved: {gguf_out}")
    else:
        warn("llama.cpp convert script not found.")
        warn("Install: git clone https://github.com/ggerganov/llama.cpp")
        print(f"  Manual: python convert_hf_to_gguf.py {final_dir} --outfile {gguf_out} --outtype {action['quant'].lower()}")

    tgt = action.get("target","ollama")
    if tgt in ("ollama","both"):
        mf = os.path.join(gguf_dir,"Modelfile")
        with open(mf,"w") as f:
            f.write(f"FROM {gguf_out}\n\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER top_k 40\nPARAMETER repeat_penalty 1.1\nPARAMETER num_ctx 2048\n\nSYSTEM \"\"\"You are Yuuki, a helpful AI assistant created by OpceanAI.\"\"\"\n")
        ok(f"Modelfile: {mf}")
        print(f"\n  {B}Ollama:{RS}")
        print(f"    ollama create {short.lower()} -f {mf}")
        print(f"    ollama run {short.lower()}")
    if tgt in ("llamacpp","both"):
        print(f"\n  {B}llama.cpp:{RS}")
        print(f"    ./llama-cli -m {gguf_out} -c 2048 -n 512 --temp 0.7 -i")

def do_upload(final_dir, action):
    sec("Post-Training  — Upload to HuggingFace")
    if not HAS_HUB: err("huggingface_hub not installed."); return
    try:
        if action.get("token"): hf_login(token=action["token"])
        api = HfApi()
        api.create_repo(action["repo"], private=action.get("private",False), exist_ok=True)
        ok(f"Uploading {final_dir} -> {action['repo']}")
        api.upload_folder(folder_path=final_dir, repo_id=action["repo"],
                          repo_type="model", commit_message="Upload fine-tuned Yuuki model")
        ok(f"Done -> https://huggingface.co/{action['repo']}")
    except Exception as e: err(f"Upload failed: {e}")


def main():
    global _trainer_ref, _tokenizer_ref, _stop_training

    banner()

    model_cfg    = w_model()
    hw_cfg       = w_hw(is_gguf=not model_cfg["model_is_hf"])
    lora_cfg     = w_lora(hw_cfg["mode"])
    hw_cfg["lora_cfg"] = lora_cfg
    ds_cfg       = w_datasets()
    hp_cfg       = w_hp()
    post_actions = w_post()

    cfg = {"model":model_cfg,"hw":hw_cfg,"ds":ds_cfg,"hp":hp_cfg,
           "post_actions":post_actions,"lora_cfg":lora_cfg}
    w_summary(cfg)

    out_dir = hp_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def _sig(sig, frame):
        global _stop_training
        logger.warning("SIGINT — saving..."); _stop_training = True
        emerg(out_dir, "sigint"); sys.exit(0)
    signal.signal(signal.SIGINT, _sig)

    if not model_cfg["model_is_hf"]:
        sec("Inference Setup Only")
        gp = model_cfg["gguf_path"]
        ok(f"GGUF: {gp}")
        gguf_dir = os.path.join(out_dir,"gguf"); os.makedirs(gguf_dir, exist_ok=True)
        for pa in post_actions:
            if pa["action"] == "gguf":
                tgt = pa.get("target","ollama")
                if tgt in ("ollama","both"):
                    mf = os.path.join(gguf_dir,"Modelfile")
                    with open(mf,"w") as f:
                        f.write(f"FROM {gp}\n\nPARAMETER temperature 0.7\nPARAMETER num_ctx 2048\n\nSYSTEM \"You are Yuuki.\"\n")
                    ok(f"Modelfile: {mf}")
                    print(f"  ollama create yuuki -f {mf}"); print(f"  ollama run yuuki")
                if tgt in ("llamacpp","both"):
                    print(f"  ./llama-cli -m {gp} -c 2048 -n 512 --temp 0.7 -i")
        return

    last_ckpt = get_last_checkpoint(out_dir)
    if last_ckpt: ok(f"Resuming from: {last_ckpt}")

    model, tokenizer, device = load_model(model_cfg, hw_cfg)
    _tokenizer_ref = tokenizer

    logger.info("Building datasets...")
    train_ds = build_data(ds_cfg, tokenizer, hp_cfg["max_length"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    mode, q = hw_cfg["mode"], hw_cfg["quantization"]
    if device=="cuda" and HAS_BNB and q=="none" and mode=="full": optim="adamw_8bit"
    elif device=="cuda" and mode in ("lora","qlora"):              optim="paged_adamw_8bit" if HAS_BNB else "adamw_torch"
    elif device=="cuda":                                           optim="adamw_torch_fused"
    else:                                                          optim="adamw_torch"

    t_args = TrainingArguments(
        output_dir=out_dir, overwrite_output_dir=False,
        num_train_epochs=hp_cfg["epochs"],
        max_steps=hp_cfg["max_steps"] if hp_cfg["max_steps"]>0 else -1,
        per_device_train_batch_size=hp_cfg["batch_size"],
        gradient_accumulation_steps=hp_cfg["grad_accum"],
        learning_rate=hp_cfg["lr"], warmup_steps=hp_cfg["warmup"],
        weight_decay=hp_cfg["weight_decay"],
        lr_scheduler_type=hp_cfg["lr_scheduler"],
        fp16=(device=="cuda" and hw_cfg["precision"]=="fp16"),
        bf16=(device=="cuda" and hw_cfg["precision"]=="bf16"),
        tf32=(device=="cuda"),
        save_strategy="steps", save_steps=hp_cfg["save_steps"],
        save_total_limit=hp_cfg["save_limit"], logging_steps=10,
        dataloader_num_workers=0, dataloader_pin_memory=(device=="cuda"),
        remove_unused_columns=False, report_to=[],
        gradient_checkpointing=(q=="none" and mode=="full"),
        optim=optim, max_grad_norm=hp_cfg["max_grad_norm"], seed=hp_cfg["seed"],
    )

    trainer = Trainer(
        model=model, args=t_args, data_collator=collator, train_dataset=train_ds,
        callbacks=[ResMonCB(), DockerCB(), CkptCB(out_dir,hp_cfg["save_limit"]), StopCB()],
    )
    _trainer_ref = trainer

    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
    except KeyboardInterrupt:
        emerg(out_dir,"keyboard"); sys.exit(0)
    except Exception as e:
        logger.exception(f"Crashed: {e}"); emerg(out_dir,"crash"); raise

    final_dir = os.path.join(out_dir,"final")
    ok(f"Saving -> {final_dir}")

    if mode in ("lora","qlora") and lora_cfg.get("merge"):
        try:
            model.merge_and_unload().save_pretrained(final_dir)
            ok("LoRA merged into base model.")
        except Exception as e:
            warn(f"Merge failed: {e}"); trainer.save_model(final_dir)
    else:
        trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    ok("Training complete.")

    for pa in post_actions:
        if pa["action"]=="gguf":       do_gguf(final_dir, model_cfg["model_path"], pa)
        elif pa["action"]=="hf_upload": do_upload(final_dir, pa)


if __name__ == "__main__":
    main()
