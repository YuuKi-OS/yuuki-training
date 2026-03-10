from __future__ import annotations
import os, logging, shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("train_yuuki.post")

C  = "\033[96m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[1m"
RS = "\033[0m"


def _find_llama_cpp_convert() -> Optional[str]:
    candidates = [
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "/usr/local/lib/python3.11/dist-packages/llama_cpp/convert_hf_to_gguf.py",
        "/usr/local/lib/python3.10/dist-packages/llama_cpp/convert_hf_to_gguf.py",
        "/opt/llama.cpp/convert_hf_to_gguf.py",
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def convert_to_gguf(final_dir: str, quant: str, target: str,
                    model_name: str = "yuuki") -> Optional[str]:
    print(f"\n{B}{C}  ─── GGUF Conversion ({quant}) ───{RS}\n")
    script = _find_llama_cpp_convert()
    gguf_dir = os.path.join(final_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    gguf_out = os.path.join(gguf_dir, f"model-{quant}.gguf")
    short    = Path(model_name).name.lower() if model_name else "yuuki"

    if script:
        print(f"  {G}✓{RS} Found convert script: {script}")
        ret = os.system(f"python {script} {final_dir} --outfile {gguf_out} --outtype {quant.lower()}")
        if ret != 0:
            print(f"  {Y}⚠{RS} Conversion failed — check llama.cpp.")
            return None
        print(f"  {G}✓{RS} GGUF saved: {gguf_out}")
    else:
        print(f"  {Y}⚠{RS} llama.cpp not found.")
        print(f"  Install: git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt")
        print(f"\n  Manual conversion:")
        print(f"    python convert_hf_to_gguf.py {final_dir} --outfile {gguf_out} --outtype {quant.lower()}")
        gguf_out = None

    if target in ("ollama", "both"):
        _generate_modelfile(gguf_dir, gguf_out or f"<path-to-{quant}.gguf>", short)

    if target in ("llamacpp", "both"):
        src = gguf_out or f"<path-to-{quant}.gguf>"
        print(f"\n  {B}llama.cpp:{RS}")
        print(f"    ./llama-cli -m {src} -c 2048 -n 512 --temp 0.7 -i")

    return gguf_out


def _generate_modelfile(gguf_dir: str, gguf_path: str, model_name: str):
    mf_path = os.path.join(gguf_dir, "Modelfile")
    content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "### Human:"
PARAMETER stop "### Assistant:"

SYSTEM \"\"\"You are Yuuki, a helpful and knowledgeable AI assistant created by OpceanAI. You are precise, friendly, and always try to give accurate and useful answers.\"\"\"

TEMPLATE \"\"\"{{ if .System }}### System:
{{ .System }}

{{ end }}{{ if .Prompt }}### Human:
{{ .Prompt }}

{{ end }}### Assistant:
{{ .Response }}\"\"\"
"""
    with open(mf_path, "w") as f:
        f.write(content)
    print(f"  {G}✓{RS} Modelfile: {mf_path}")
    print(f"\n  {B}Ollama:{RS}")
    print(f"    ollama create {model_name} -f {mf_path}")
    print(f"    ollama run {model_name}")


def upload_to_hf(final_dir: str, repo: str, token: Optional[str],
                 private: bool, webhook_notifier=None):
    print(f"\n{B}{C}  ─── HuggingFace Upload ───{RS}\n")
    try:
        from huggingface_hub import HfApi, login as hf_login
    except ImportError:
        print(f"  {Y}⚠{RS} huggingface_hub not installed.")
        return

    try:
        if token:
            hf_login(token=token)
        api = HfApi()
        api.create_repo(repo, private=private, exist_ok=True)
        print(f"  {G}✓{RS} Uploading {final_dir} → {repo}")

        api.upload_folder(
            folder_path=final_dir,
            repo_id=repo,
            repo_type="model",
            commit_message="Upload fine-tuned Yuuki NxG model",
        )
        print(f"  {G}✓{RS} Done → https://huggingface.co/{repo}")

        if webhook_notifier:
            webhook_notifier.notify_upload(repo)

    except Exception as e:
        print(f"  {Y}⚠{RS} Upload failed: {e}")


def generate_readme(final_dir: str, cfg) -> str:
    model_name = cfg.model.model_path or "Yuuki NxG"
    repo_name  = cfg.post.hf_repo or "OpceanAI/Yuuki-finetuned"
    datasets   = ", ".join(d.label for d in cfg.dataset.datasets)
    mode_str   = cfg.hw.mode.upper()

    readme = f"""---
license: apache-2.0
base_model: {model_name}
tags:
- yuuki
- opceanai
- fine-tuned
- causal-lm
---

# {repo_name.split('/')[-1]}

Fine-tuned version of [{model_name}](https://huggingface.co/{model_name}).

## Training Details

| Setting | Value |
|---------|-------|
| Base model | `{model_name}` |
| Training mode | {mode_str} |
| Precision | {cfg.hw.precision} |
| Quantization | {cfg.hw.quantization} |
| Datasets | {datasets} |
| Learning rate | {cfg.hp.lr} |
| Epochs | {cfg.hp.epochs} |
| Seq length | {cfg.hp.max_length} |
| Batch (effective) | {cfg.hp.batch_size * cfg.hp.grad_accum} |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

prompt = "### Human:\\nTell me something interesting.\\n\\n### Assistant:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Trained with [Yuuki Trainer](https://github.com/YuuKi-OS/yuuki-trainer)
"""

    readme_path = os.path.join(final_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    return readme_path


def run_post_training(final_dir: str, cfg, tokenizer=None, webhook=None):
    model_name = (cfg.model.model_path or "yuuki").split("/")[-1]

    if cfg.post.chat_test:
        from core.model_utils import run_chat_test
        run_chat_test(final_dir, tokenizer)

    if cfg.post.convert_gguf:
        convert_to_gguf(
            final_dir=final_dir,
            quant=cfg.post.gguf_quant,
            target=cfg.post.gguf_target,
            model_name=model_name,
        )

    if cfg.post.convert_onnx:
        from core.model_utils import export_to_onnx
        onnx_dir = os.path.join(final_dir, "onnx")
        export_to_onnx(final_dir, onnx_dir, tokenizer, cfg.hp.max_length)

    if cfg.post.convert_gptq:
        from core.model_utils import convert_to_gptq
        gptq_dir = os.path.join(final_dir, "gptq")
        convert_to_gptq(final_dir, gptq_dir, cfg.post.gptq_bits)

    if cfg.post.upload_hf and cfg.post.hf_repo:
        generate_readme(final_dir, cfg)
        upload_to_hf(
            final_dir=final_dir,
            repo=cfg.post.hf_repo,
            token=cfg.post.hf_token or None,
            private=cfg.post.hf_private,
            webhook_notifier=webhook,
        )
