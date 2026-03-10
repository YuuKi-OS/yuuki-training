#!/usr/bin/env python3
"""
Yuuki Trainer Web UI  —  OpceanAI
Run: python web_ui.py
"""

import os, sys, json, time, threading, queue
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

HAS_TORCH = False
try:
    import torch; HAS_TORCH = True
except: pass

DARK_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Syne"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#09090b",
    body_background_fill_dark="#09090b",
    block_background_fill="#18181b",
    block_background_fill_dark="#18181b",
    block_border_color="#27272a",
    block_border_color_dark="#27272a",
    block_label_background_fill="#18181b",
    block_label_background_fill_dark="#18181b",
    block_label_text_color="#a1a1aa",
    block_label_text_color_dark="#a1a1aa",
    input_background_fill="#09090b",
    input_background_fill_dark="#09090b",
    input_border_color="#27272a",
    input_border_color_dark="#27272a",
    input_placeholder_color="#52525b",
    input_placeholder_color_dark="#52525b",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#7c3aed",
    button_secondary_background_fill="#27272a",
    button_secondary_background_fill_dark="#27272a",
    button_secondary_text_color="#e4e4e7",
    button_secondary_border_color="#3f3f46",
    body_text_color="#e4e4e7",
    body_text_color_dark="#e4e4e7",
    block_title_text_color="#f4f4f5",
    block_title_text_color_dark="#f4f4f5",
    checkbox_background_color="#27272a",
    checkbox_background_color_dark="#27272a",
    slider_color="#7c3aed",
    table_even_background_fill="#18181b",
    table_odd_background_fill="#09090b",
    table_border_color="#27272a",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-base: #09090b;
  --bg-surface: #18181b;
  --bg-elevated: #27272a;
  --border: #3f3f46;
  --border-subtle: #27272a;
  --text-primary: #f4f4f5;
  --text-secondary: #a1a1aa;
  --text-muted: #71717a;
  --accent: #7c3aed;
  --accent-light: #a78bfa;
  --accent-glow: rgba(124, 58, 237, 0.15);
  --green: #22c55e;
  --yellow: #eab308;
  --red: #ef4444;
  --radius: 10px;
}

* { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
}

.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 0 16px !important;
}

/* Header */
.yuuki-header {
  padding: 48px 0 32px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 32px;
}
.yuuki-logo {
  font-family: 'Syne', sans-serif;
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -2px;
  background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 50%, #6d28d9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
  margin: 0;
}
.yuuki-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: var(--text-muted);
  margin: 8px 0 0;
  letter-spacing: 0.05em;
}
.yuuki-badge {
  display: inline-block;
  background: var(--accent-glow);
  border: 1px solid rgba(124,58,237,0.3);
  color: var(--accent-light);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 100px;
  margin-top: 12px;
  letter-spacing: 0.08em;
}

/* Cards */
.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 16px;
}
.card-title {
  font-family: 'Syne', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-title::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 14px;
  background: var(--accent);
  border-radius: 2px;
}

/* Tabs */
.tab-nav {
  border-bottom: 1px solid var(--border-subtle) !important;
  background: transparent !important;
  margin-bottom: 24px !important;
}
.tab-nav button {
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 12px 20px !important;
  border-radius: 0 !important;
  transition: all 0.2s !important;
}
.tab-nav button.selected {
  color: var(--text-primary) !important;
  border-bottom-color: var(--accent) !important;
}
.tab-nav button:hover:not(.selected) {
  color: var(--text-secondary) !important;
}

/* Form elements */
input, textarea, select {
  background: var(--bg-base) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
  transition: border-color 0.2s !important;
}
input:focus, textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
label {
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
  letter-spacing: 0.02em !important;
}

/* Buttons */
button.primary {
  background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
  border: none !important;
  border-radius: 8px !important;
  color: #fff !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.03em !important;
  padding: 12px 28px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
}
button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(124,58,237,0.45) !important;
}
button.secondary {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-primary) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

/* Progress bar */
.progress-wrap {
  background: var(--bg-base);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 16px 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}
.progress-bar-outer {
  width: 100%;
  height: 6px;
  background: var(--bg-elevated);
  border-radius: 3px;
  margin: 12px 0 8px;
  overflow: hidden;
}
.progress-bar-inner {
  height: 100%;
  background: linear-gradient(90deg, #7c3aed, #a78bfa);
  border-radius: 3px;
  transition: width 0.3s ease;
  box-shadow: 0 0 12px rgba(124,58,237,0.6);
}
.progress-meta {
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
  font-size: 12px;
}

/* Log output */
.log-box {
  background: #050507 !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  color: #86efac !important;
  padding: 16px !important;
  max-height: 340px !important;
  overflow-y: auto !important;
  line-height: 1.6 !important;
}
.log-box::before {
  content: '$ ';
  color: #7c3aed;
}

/* Status chips */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 100px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 500;
}
.chip-idle    { background: rgba(161,161,170,0.1); color: #a1a1aa; border: 1px solid rgba(161,161,170,0.2); }
.chip-running { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.chip-done    { background: rgba(34,197,94,0.1);   color: #86efac; border: 1px solid rgba(34,197,94,0.2); }
.chip-error   { background: rgba(239,68,68,0.1);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }

/* Stat cards */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin: 16px 0;
}
.stat-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}
.stat-value {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  color: var(--accent-light);
  line-height: 1;
}
.stat-label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
}

/* Tooltip style info boxes */
.info-box {
  background: rgba(124,58,237,0.07);
  border: 1px solid rgba(124,58,237,0.2);
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 13px;
  color: var(--text-secondary);
  margin: 8px 0;
}
.info-box strong { color: var(--accent-light); }

/* Chat messages */
.message-user {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 12px 12px 4px 12px;
  padding: 12px 16px;
  margin: 8px 0;
  margin-left: 20%;
  font-size: 14px;
}
.message-bot {
  background: rgba(124,58,237,0.08);
  border: 1px solid rgba(124,58,237,0.15);
  border-radius: 12px 12px 12px 4px;
  padding: 12px 16px;
  margin: 8px 0;
  margin-right: 20%;
  font-size: 14px;
}

/* Dropdown */
.gradio-dropdown .wrap {
  background: var(--bg-base) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* Slider */
.gradio-slider input[type=range] {
  accent-color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #52525b; }

/* Accordion */
.gradio-accordion > .label-wrap {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  color: var(--text-secondary) !important;
}

/* Hide default gradio footer */
footer { display: none !important; }
.gradio-container > .footer { display: none !important; }

/* Responsive */
@media (max-width: 768px) {
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
  .gradio-container { padding: 0 8px !important; }
}

/* Animation */
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(124,58,237,0.3); }
  50%       { box-shadow: 0 0 20px rgba(124,58,237,0.6); }
}
.running-indicator {
  animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.spinner {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
"""

_log_queue   = queue.Queue(maxsize=500)
_stop_flag   = [False]
_train_state = {"running": False, "step": 0, "total": 0, "loss": None, "eta": ""}


def detect_gpu_info() -> str:
    if not HAS_TORCH:
        return "⬜ No PyTorch — CPU only"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        count = torch.cuda.device_count()
        multi = f"  ({count}× GPUs)" if count > 1 else ""
        return f"🟢 CUDA — {name}  {vram} GB{multi}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "🟡 Apple MPS"
    return "⬜ CPU only"


def get_hf_datasets(query: str) -> list:
    if not query or len(query) < 2:
        return []
    try:
        from huggingface_hub import HfApi
        results = list(HfApi().list_datasets(search=query, limit=10))
        return [r.id for r in results]
    except Exception:
        return []


def get_hf_models(query: str) -> list:
    if not query or len(query) < 2:
        return []
    try:
        from huggingface_hub import HfApi
        results = list(HfApi().list_models(search=query, limit=10, filter="text-generation"))
        return [r.id for r in results]
    except Exception:
        return []


def estimate_resources(model_id, mode, precision, batch, grad_accum, seq_len, epochs):
    try:
        from core.config import (
            TrainingConfig, ModelConfig, HardwareConfig, HyperParams, DatasetConfig, DatasetEntry
        )
        from core.estimator import build_estimate_report
        cfg = TrainingConfig(
            model=ModelConfig(model_path=model_id),
            hw=HardwareConfig(mode=mode, use_gpu=True, precision=precision),
            hp=HyperParams(batch_size=batch, grad_accum=grad_accum,
                           max_length=seq_len, epochs=epochs),
            dataset=DatasetConfig(datasets=[
                DatasetEntry(type="hf", id="OpceanAI/Yuuki-dataset", label="Yuuki dataset")
            ]),
        )
        return build_estimate_report(cfg)
    except Exception as e:
        return f"Estimate unavailable: {e}"


def build_config_from_ui(
    model_id, model_mode, train_mode, precision, quantization,
    flash_attn, compile_model, multi_gpu,
    lora_r, lora_alpha, lora_dropout, lora_targets, lora_merge,
    datasets_text, local_files, buffer_size, val_ratio, preview, extend_tok,
    output_dir, max_length, batch_size, grad_accum, epochs, max_steps,
    lr, warmup, weight_decay, max_grad_norm, lr_scheduler, save_steps, save_limit, seed,
    use_wandb, wandb_project, wandb_key,
    webhook_enabled, webhook_url, webhook_platform,
    dpo_enabled, dpo_dataset, dpo_beta,
    optuna_enabled, optuna_trials,
    post_chat, post_gguf, post_gguf_quant, post_gguf_target,
    post_onnx, post_gptq, post_gptq_bits,
    post_upload, post_hf_repo, post_hf_token, post_hf_private,
    post_synthetic, post_synthetic_n,
):
    from core.config import (
        TrainingConfig, ModelConfig, HardwareConfig, LoraConfig,
        DatasetConfig, DatasetEntry, HyperParams, LoggingConfig,
        WebhookConfig, PostTrainingConfig, DPOConfig, OptimizationConfig,
    )

    datasets = []
    for line in (datasets_text or "").strip().split("\n"):
        line = line.strip()
        if line:
            parts = line.split("|")
            ds_id = parts[0].strip()
            split = parts[1].strip() if len(parts) > 1 else "train"
            datasets.append(DatasetEntry(type="hf", id=ds_id, split=split, label=ds_id))

    if local_files:
        for fpath in local_files:
            datasets.append(DatasetEntry(type="local", path=fpath, label=Path(fpath).name))

    if not datasets:
        datasets = [
            DatasetEntry(type="hf", id="OpceanAI/Yuuki-dataset", split="train", label="Yuuki dataset"),
        ]

    modules = [m.strip() for m in lora_targets.split(",")] if lora_targets.strip() else None

    report_to = []
    if use_wandb: report_to.append("wandb")

    cfg = TrainingConfig(
        model=ModelConfig(model_path=model_id, model_is_hf=True),
        hw=HardwareConfig(
            mode=train_mode.lower(), use_gpu=True,
            precision=precision.lower(), quantization=quantization.lower(),
            flash_attn=flash_attn, compile=compile_model, multi_gpu=multi_gpu,
        ),
        lora=LoraConfig(r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                        target_modules=modules, merge=lora_merge),
        dataset=DatasetConfig(
            datasets=datasets, buffer=buffer_size,
            extend_tokenizer=extend_tok, preview_before_train=preview,
            val_split_ratio=val_ratio,
        ),
        hp=HyperParams(
            output_dir=output_dir, max_length=max_length,
            batch_size=batch_size, grad_accum=grad_accum,
            epochs=epochs, max_steps=max_steps,
            lr=lr, warmup=warmup, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm, lr_scheduler=lr_scheduler,
            save_steps=save_steps, save_limit=save_limit, seed=seed,
        ),
        logging=LoggingConfig(
            use_wandb=use_wandb, wandb_project=wandb_project or "yuuki-nxg",
            wandb_api_key=wandb_key or "",
        ),
        webhook=WebhookConfig(
            enabled=webhook_enabled, url=webhook_url or "",
            platform=webhook_platform.lower() if webhook_platform else "discord",
            on_finish=True, on_crash=True,
        ),
        dpo=DPOConfig(
            enabled=dpo_enabled, dataset_id=dpo_dataset or "",
            beta=dpo_beta,
        ),
        optim=OptimizationConfig(use_optuna=optuna_enabled, optuna_trials=optuna_trials),
        post=PostTrainingConfig(
            chat_test=False,
            convert_gguf=post_gguf, gguf_quant=post_gguf_quant, gguf_target=post_gguf_target.lower(),
            convert_onnx=post_onnx,
            convert_gptq=post_gptq, gptq_bits=post_gptq_bits,
            upload_hf=post_upload, hf_repo=post_hf_repo or "",
            hf_token=post_hf_token or "", hf_private=post_hf_private,
            synthetic_data=post_synthetic, synthetic_samples=post_synthetic_n,
        ),
    )
    return cfg


def launch_training_thread(cfg):
    import subprocess, sys
    _stop_flag[0]      = False
    _train_state.update({"running": True, "step": 0, "total": 0, "loss": None, "eta": ""})

    cfg_path = os.path.join(cfg.hp.output_dir, "training_config.json")
    os.makedirs(cfg.hp.output_dir, exist_ok=True)
    cfg.save(cfg_path)
    _log_queue.put(f"Config saved: {cfg_path}\n")

    def run():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "train", os.path.join(os.path.dirname(__file__), "train.py")
            )
            mod = importlib.util.load_from_spec(spec)
            spec.loader.exec_module(mod)

            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                try:
                    mod.run_training(cfg)
                except SystemExit:
                    pass
            _log_queue.put(buf.getvalue())
        except Exception as e:
            _log_queue.put(f"ERROR: {e}\n")
        finally:
            _train_state["running"] = False
            _log_queue.put("__DONE__\n")

    t = threading.Thread(target=run, daemon=True)
    t.start()


def collect_logs() -> str:
    lines = []
    try:
        while True:
            item = _log_queue.get_nowait()
            if isinstance(item, dict):
                if "step" in item:
                    _train_state.update({
                        "step": item.get("step",0),
                        "total": item.get("total",0),
                        "loss": item.get("loss"),
                        "eta": item.get("eta",""),
                    })
            else:
                lines.append(str(item))
    except queue.Empty:
        pass
    return "".join(lines)


def make_progress_html() -> str:
    s = _train_state
    step, total = s["step"], s["total"]
    loss = s["loss"]
    eta  = s["eta"]
    pct  = round(step/total*100, 1) if total > 0 else 0

    status_cls  = "chip-running" if s["running"] else ("chip-done" if step > 0 else "chip-idle")
    status_text = "Training..." if s["running"] else ("Complete" if step > 0 else "Idle")
    spinner     = '<span class="spinner"></span>' if s["running"] else ""

    loss_str = f"Loss: {loss:.4f}" if loss else ""
    eta_str  = f"ETA: {eta}" if eta else ""

    return f"""
<div class="progress-wrap running-indicator" style="{'animation:none' if not s['running'] else ''}">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
    <span class="chip {status_cls}">{spinner} {status_text}</span>
    <span style="font-size:12px; color:#a1a1aa;">Step {step:,} / {total:,}</span>
  </div>
  <div class="progress-bar-outer">
    <div class="progress-bar-inner" style="width:{pct}%"></div>
  </div>
  <div class="progress-meta">
    <span>{loss_str}</span>
    <span style="color:#a78bfa; font-weight:700;">{pct}%</span>
    <span>{eta_str}</span>
  </div>
</div>
"""


def chat_with_model(model_path: str, history: list, user_msg: str, temp: float, max_tok: int):
    if not user_msg.strip():
        return history, ""
    history = history or []
    history.append([user_msg, None])

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16 if HAS_TORCH and torch.cuda.is_available() else torch.float32,
            device_map="auto" if HAS_TORCH and torch.cuda.is_available() else None,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = f"### Human:\n{user_msg}\n\n### Assistant:\n"
        result = pipe(prompt, max_new_tokens=max_tok, temperature=temp,
                       top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = result[0]["generated_text"][len(prompt):]
        history[-1][1] = response.strip()
    except Exception as e:
        history[-1][1] = f"⚠ Error: {e}"

    return history, ""


def build_ui():
    gpu_info = detect_gpu_info()

    with gr.Blocks(theme=DARK_THEME, css=CSS, title="Yuuki Trainer") as demo:

        gr.HTML(f"""
        <div class="yuuki-header">
          <h1 class="yuuki-logo">YUUKI</h1>
          <p class="yuuki-subtitle">NxG Training Wizard  ·  OpceanAI</p>
          <span class="yuuki-badge">v2.0  ·  {gpu_info}</span>
        </div>
        """)

        with gr.Tabs(elem_classes="tab-nav"):

            # ─── TAB 1: MODEL ───────────────────────────────────────
            with gr.TabItem("⬡ Model"):
                gr.HTML('<div class="card-title">Base Model</div>')

                with gr.Row():
                    model_id = gr.Textbox(
                        value="OpceanAI/Yuuki-NxG",
                        label="HuggingFace Model ID or local path",
                        placeholder="e.g. OpceanAI/Yuuki-NxG  or  ./models/yuuki",
                    )
                    model_search_btn = gr.Button("🔍 Search HF", variant="secondary", scale=0)

                model_search_results = gr.Dropdown(
                    label="Search Results  (click to select)",
                    choices=[], visible=False, interactive=True,
                )
                model_search_query = gr.Textbox(
                    label="Search HuggingFace models",
                    placeholder="Type to search — e.g. 'llama 3b'",
                )

                gr.HTML("""
                <div class="info-box">
                  <strong>Supported formats:</strong> HuggingFace ID · Local directory · .pt · .safetensors
                  <br><strong>Recommended:</strong> OpceanAI/Yuuki-NxG (3B) or OpceanAI/Yuuki-NxG-Nano (81M)
                </div>
                """)

                model_mode = gr.Radio(
                    ["HuggingFace Hub", "Local Directory"],
                    value="HuggingFace Hub",
                    label="Source",
                )

            # ─── TAB 2: TRAINING ────────────────────────────────────
            with gr.TabItem("⚡ Training"):
                with gr.Row():
                    train_mode = gr.Dropdown(
                        ["lora", "qlora", "full"],
                        value="lora",
                        label="Training Mode",
                    )
                    precision = gr.Dropdown(
                        ["fp16", "bf16", "fp32"],
                        value="fp16",
                        label="Precision",
                    )
                    quantization = gr.Dropdown(
                        ["none", "int8", "int4"],
                        value="none",
                        label="Base Quantization",
                    )

                with gr.Row():
                    flash_attn    = gr.Checkbox(False, label="Flash Attention 2")
                    compile_model = gr.Checkbox(False, label="torch.compile")
                    multi_gpu     = gr.Checkbox(False, label="Multi-GPU (accelerate)")

                with gr.Accordion("LoRA / QLoRA Settings", open=True):
                    with gr.Row():
                        lora_r       = gr.Slider(1, 128, value=16, step=1, label="Rank (r)")
                        lora_alpha   = gr.Slider(1, 256, value=32, step=1, label="Alpha")
                        lora_dropout = gr.Slider(0, 0.5, value=0.05, step=0.01, label="Dropout")
                    lora_targets = gr.Textbox(
                        value="q_proj,v_proj,k_proj,o_proj",
                        label="Target Modules  (comma-separated, empty = auto)",
                    )
                    lora_merge = gr.Checkbox(False, label="Merge adapters after training")

                gr.HTML("""
                <div class="info-box" id="mode-help">
                  <strong>Full</strong>: All params updated — needs max VRAM.
                  <strong>LoRA</strong>: Only small adapters — recommended for &lt;16GB VRAM.
                  <strong>QLoRA</strong>: LoRA + 4-bit base — fits in 6GB VRAM.
                </div>
                """)

            # ─── TAB 3: DATASETS ────────────────────────────────────
            with gr.TabItem("📦 Datasets"):
                gr.HTML('<div class="info-box">One dataset per line: <strong>org/dataset-id</strong>  or  <strong>org/dataset-id | split</strong></div>')

                datasets_text = gr.Textbox(
                    value="OpceanAI/Yuuki-dataset | train\nscryptiam/anime-waifu-personality-chat | train\nTuringEnterprises/Open-RL | train",
                    label="HuggingFace Datasets",
                    lines=6,
                    placeholder="org/dataset | split",
                )

                with gr.Row():
                    ds_search_q   = gr.Textbox(label="Search HF Datasets", placeholder="Search...")
                    ds_search_btn = gr.Button("🔍", variant="secondary", scale=0)

                ds_search_results = gr.Dropdown(label="Results", choices=[], visible=False)

                local_files = gr.File(
                    label="Upload local files  (.json .jsonl .csv)",
                    file_count="multiple",
                    file_types=[".json", ".jsonl", ".csv"],
                )

                with gr.Row():
                    buffer_size = gr.Slider(100, 10000, value=1000, step=100, label="Shuffle Buffer")
                    val_ratio   = gr.Slider(0, 0.3, value=0.05, step=0.01, label="Validation Split")

                with gr.Row():
                    preview    = gr.Checkbox(True,  label="Preview examples before training")
                    extend_tok = gr.Checkbox(False, label="Extend tokenizer vocabulary")

            # ─── TAB 4: HYPERPARAMS ────────────────────────────────
            with gr.TabItem("🎛 Hyperparameters"):
                output_dir = gr.Textbox(value="./yuuki_output", label="Output Directory")

                with gr.Row():
                    max_length = gr.Slider(64, 8192, value=512, step=64, label="Max Sequence Length")
                    seed       = gr.Slider(0, 9999, value=42, step=1, label="Random Seed")

                with gr.Row():
                    batch_size = gr.Slider(1, 32, value=1, step=1, label="Batch Size")
                    grad_accum = gr.Slider(1, 64, value=8, step=1, label="Gradient Accumulation")

                effective_batch_md = gr.Markdown("**Effective batch:** 8")

                with gr.Row():
                    epochs    = gr.Slider(1, 50, value=3, step=1, label="Epochs")
                    max_steps = gr.Number(value=-1, label="Max Steps  (-1 = unlimited)")

                with gr.Row():
                    lr           = gr.Number(value=2e-5, label="Learning Rate", precision=8)
                    warmup       = gr.Slider(0, 1000, value=100, step=10, label="Warmup Steps")
                    weight_decay = gr.Number(value=0.01, label="Weight Decay", precision=4)

                with gr.Row():
                    max_grad_norm = gr.Number(value=1.0, label="Max Grad Norm", precision=3)
                    lr_scheduler  = gr.Dropdown(
                        ["cosine","linear","constant","cosine_with_restarts"],
                        value="cosine", label="LR Scheduler",
                    )

                with gr.Row():
                    save_steps = gr.Slider(10, 1000, value=100, step=10, label="Save Every N Steps")
                    save_limit = gr.Slider(1, 20, value=3, step=1, label="Max Checkpoints")

                def update_effective(bs, ga):
                    return f"**Effective batch:** {int(bs) * int(ga)}"
                batch_size.change(update_effective, [batch_size, grad_accum], effective_batch_md)
                grad_accum.change(update_effective, [batch_size, grad_accum], effective_batch_md)

            # ─── TAB 5: ADVANCED ───────────────────────────────────
            with gr.TabItem("🔬 Advanced"):

                with gr.Accordion("Weights & Biases / TensorBoard", open=False):
                    with gr.Row():
                        use_wandb     = gr.Checkbox(False, label="Enable W&B")
                        wandb_project = gr.Textbox(value="yuuki-nxg", label="W&B Project")
                        wandb_key     = gr.Textbox(value="", label="W&B API Key", type="password")

                with gr.Accordion("Webhook Notifications", open=False):
                    with gr.Row():
                        webhook_enabled  = gr.Checkbox(False, label="Enable Webhooks")
                        webhook_platform = gr.Dropdown(
                            ["Discord","Slack","Telegram","ntfy","Custom"],
                            value="Discord", label="Platform",
                        )
                    webhook_url = gr.Textbox(value="", label="Webhook URL")

                with gr.Accordion("DPO / Alignment Training", open=False):
                    dpo_enabled = gr.Checkbox(False, label="Enable DPO")
                    gr.HTML('<div class="info-box">Needs dataset with <strong>prompt</strong>, <strong>chosen</strong>, <strong>rejected</strong> columns.</div>')
                    with gr.Row():
                        dpo_dataset = gr.Textbox(value="Anthropic/hh-rlhf", label="DPO Dataset ID")
                        dpo_beta    = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Beta")

                with gr.Accordion("Optuna Hyperparameter Search", open=False):
                    optuna_enabled = gr.Checkbox(False, label="Run Optuna search before training")
                    optuna_trials  = gr.Slider(2, 50, value=10, step=1, label="Number of Trials")
                    gr.HTML('<div class="info-box">Runs N short training trials to find optimal LR and batch size before the full run.</div>')

            # ─── TAB 6: POST-TRAINING ──────────────────────────────
            with gr.TabItem("📤 Post-Training"):

                with gr.Accordion("GGUF Conversion  (Ollama / llama.cpp)", open=True):
                    post_gguf = gr.Checkbox(False, label="Convert to GGUF after training")
                    with gr.Row():
                        post_gguf_quant  = gr.Dropdown(
                            ["Q4_K_M","Q5_K_M","Q8_0","f16","f32"],
                            value="Q4_K_M", label="Quantization",
                        )
                        post_gguf_target = gr.Dropdown(
                            ["Ollama","llama.cpp","Both"],
                            value="Ollama", label="Target Runtime",
                        )

                with gr.Accordion("ONNX Export  (fast CPU inference)", open=False):
                    post_onnx = gr.Checkbox(False, label="Export to ONNX")

                with gr.Accordion("GPTQ Quantization  (GPU deployment)", open=False):
                    post_gptq      = gr.Checkbox(False, label="Convert to GPTQ")
                    post_gptq_bits = gr.Slider(2, 8, value=4, step=2, label="GPTQ Bits")

                with gr.Accordion("Synthetic Data Pre-Training", open=False):
                    post_synthetic   = gr.Checkbox(False, label="Generate synthetic training data first")
                    post_synthetic_n = gr.Slider(100, 10000, value=1000, step=100, label="Number of Samples")

                with gr.Accordion("Upload to HuggingFace Hub", open=False):
                    post_upload    = gr.Checkbox(False, label="Upload after training")
                    post_hf_repo   = gr.Textbox(value="", label="Destination Repo  (e.g. YourOrg/Yuuki-ft)")
                    post_hf_token  = gr.Textbox(value="", label="Write Token", type="password")
                    post_hf_private = gr.Checkbox(False, label="Private repository")

            # ─── TAB 7: TRAINING MONITOR ──────────────────────────
            with gr.TabItem("▶ Train"):
                estimate_output = gr.Textbox(
                    label="Resource Estimate",
                    lines=7, interactive=False,
                    elem_classes="log-box",
                )
                estimate_btn = gr.Button("📊 Estimate Resources", variant="secondary")

                gr.HTML('<div style="height:16px;"></div>')

                progress_html = gr.HTML(make_progress_html())
                log_output    = gr.Textbox(
                    label="Training Log",
                    lines=16, interactive=False,
                    elem_classes="log-box",
                    autoscroll=True,
                )

                with gr.Row():
                    start_btn = gr.Button("▶  Start Training", variant="primary", scale=2)
                    stop_btn  = gr.Button("⏹  Stop", variant="secondary", scale=0)

                training_log_state = gr.State("")

                def do_estimate(mid, tm, prec, bs, ga, sl, ep):
                    return estimate_resources(mid, tm, prec, int(bs), int(ga), int(sl), int(ep))

                estimate_btn.click(
                    do_estimate,
                    inputs=[model_id, train_mode, precision, batch_size, grad_accum, max_length, epochs],
                    outputs=estimate_output,
                )

                all_inputs = [
                    model_id, model_mode, train_mode, precision, quantization,
                    flash_attn, compile_model, multi_gpu,
                    lora_r, lora_alpha, lora_dropout, lora_targets, lora_merge,
                    datasets_text, local_files, buffer_size, val_ratio, preview, extend_tok,
                    output_dir, max_length, batch_size, grad_accum, epochs, max_steps,
                    lr, warmup, weight_decay, max_grad_norm, lr_scheduler, save_steps, save_limit, seed,
                    use_wandb, wandb_project, wandb_key,
                    webhook_enabled, webhook_url, webhook_platform,
                    dpo_enabled, dpo_dataset, dpo_beta,
                    optuna_enabled, optuna_trials,
                    gr.State(True),
                    post_gguf, post_gguf_quant, post_gguf_target,
                    post_onnx, post_gptq, post_gptq_bits,
                    post_upload, post_hf_repo, post_hf_token, post_hf_private,
                    post_synthetic, post_synthetic_n,
                ]

                def start_training(*args):
                    local_files_arg = args[13]
                    local_paths = []
                    if local_files_arg:
                        for f in local_files_arg:
                            if hasattr(f, "name"):
                                local_paths.append(f.name)
                            elif isinstance(f, str):
                                local_paths.append(f)

                    new_args = list(args)
                    new_args[13] = args[12]
                    new_args[14] = local_paths

                    try:
                        cfg = build_config_from_ui(*new_args[:55])
                        launch_training_thread(cfg)
                        return "Training started! Monitor the log below.\n"
                    except Exception as e:
                        return f"Error: {e}\n"

                def stop_training():
                    _stop_flag[0] = True
                    return "Stop requested — will halt after current step.\n"

                def tick(current_log):
                    new_text = collect_logs()
                    updated  = current_log + new_text
                    if len(updated) > 40000:
                        updated = updated[-40000:]
                    return updated, make_progress_html()

                start_btn.click(start_training, inputs=all_inputs, outputs=log_output)
                stop_btn.click(stop_training, outputs=log_output)

                timer = gr.Timer(value=1.0)
                timer.tick(tick, inputs=training_log_state, outputs=[log_output, progress_html])

            # ─── TAB 8: CHAT TEST ─────────────────────────────────
            with gr.TabItem("💬 Chat Test"):
                gr.HTML('<div class="info-box">Test your trained model directly in the browser after training completes.</div>')

                with gr.Row():
                    chat_model_path = gr.Textbox(
                        value="./yuuki_output/final",
                        label="Model Path",
                        scale=3,
                    )
                    with gr.Column(scale=1):
                        chat_temp    = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                        chat_max_tok = gr.Slider(64, 1024, value=256, step=32, label="Max New Tokens")

                chatbot = gr.Chatbot(
                    label="Yuuki",
                    height=420,
                    bubble_full_width=False,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message",
                        placeholder="Talk to Yuuki...",
                        scale=4,
                    )
                    chat_send  = gr.Button("Send →", variant="primary", scale=0)
                    chat_clear = gr.Button("Clear", variant="secondary", scale=0)

                chat_state = gr.State([])

                chat_send.click(
                    chat_with_model,
                    inputs=[chat_model_path, chat_state, chat_input, chat_temp, chat_max_tok],
                    outputs=[chatbot, chat_input],
                ).then(lambda h: h, chat_state, chat_state)

                chat_input.submit(
                    chat_with_model,
                    inputs=[chat_model_path, chat_state, chat_input, chat_temp, chat_max_tok],
                    outputs=[chatbot, chat_input],
                )
                chat_clear.click(lambda: ([], []), outputs=[chatbot, chat_state])

            # ─── TAB 9: CONFIG ─────────────────────────────────────
            with gr.TabItem("⚙ Config"):
                gr.HTML('<div class="info-box">Save and load training configs as JSON — reuse previous runs or share settings.</div>')

                config_json_view = gr.Code(
                    language="json",
                    label="Current Config JSON",
                    lines=30,
                )
                with gr.Row():
                    save_config_btn = gr.Button("💾 Export Config", variant="secondary")
                    load_config_file = gr.File(label="Load Config", file_types=[".json"])

                config_save_path = gr.Textbox(value="./yuuki_config.json", label="Save Path")
                config_status    = gr.Textbox(label="Status", interactive=False)

        # ─── Model search wiring ───────────────────────────────────
        def search_models(q):
            results = get_hf_models(q)
            return gr.Dropdown(choices=results, visible=bool(results))

        def search_datasets(q):
            results = get_hf_datasets(q)
            return gr.Dropdown(choices=results, visible=bool(results))

        model_search_btn.click(search_models, inputs=model_search_query, outputs=model_search_results)
        model_search_results.change(lambda x: x, inputs=model_search_results, outputs=model_id)
        ds_search_btn.click(search_datasets, inputs=ds_search_query, outputs=ds_search_results)

        def append_dataset(ds_id, current):
            if not ds_id: return current
            return current.rstrip() + f"\n{ds_id} | train"
        ds_search_results.change(append_dataset, inputs=[ds_search_results, datasets_text], outputs=datasets_text)

        gr.HTML("""
        <div style="text-align:center; padding:32px 0 16px; color:#3f3f46; font-family:'JetBrains Mono',monospace; font-size:12px;">
          Yuuki Trainer v2.0  ·  OpceanAI  ·  <a href="https://huggingface.co/OpceanAI" style="color:#7c3aed; text-decoration:none;">huggingface.co/OpceanAI</a>
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
