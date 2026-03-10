from __future__ import annotations
import logging, os, math
from typing import Tuple, Optional, Any

logger = logging.getLogger("train_yuuki.model")

HAS_TORCH = False
HAS_BNB   = False
HAS_PEFT  = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass
try:
    import bitsandbytes
    HAS_BNB = True
except ImportError:
    pass
try:
    import peft
    HAS_PEFT = True
except ImportError:
    pass


def get_device(hw_cfg) -> str:
    if not hw_cfg.use_gpu or not HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_free_vram_gb() -> float:
    try:
        if HAS_TORCH and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            reserved = torch.cuda.memory_reserved(0)
            free = props.total_memory - reserved
            return free / 1e9
    except Exception:
        pass
    return 0.0


def should_use_gradient_checkpointing(hw_cfg, param_count: float) -> bool:
    if not hw_cfg.smart_gradient_checkpointing:
        return False
    free_vram = get_free_vram_gb()
    model_size_gb = (param_count * 2) / 1e9
    if free_vram > 0 and free_vram < model_size_gb * 1.5:
        logger.info(f"Smart gradient checkpointing: VRAM tight ({free_vram:.1f}GB free vs {model_size_gb:.1f}GB model) — enabling.")
        return True
    if param_count > 3e9:
        logger.info("Smart gradient checkpointing: large model detected — enabling.")
        return True
    return False


def build_bnb_config(hw_cfg):
    from transformers import BitsAndBytesConfig
    q    = hw_cfg.quantization
    prec = hw_cfg.precision

    if q == "int4" or hw_cfg.mode == "qlora":
        compute_dtype = (torch.bfloat16 if prec == "bf16" else torch.float16) if HAS_TORCH else None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if q == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(model_cfg, hw_cfg) -> Tuple[Any, Any, str]:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    mp = model_cfg.model_path
    logger.info(f"Loading tokenizer: {mp}")
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device(hw_cfg)
    kw = {"trust_remote_code": True}

    if device == "cuda":
        kw["device_map"] = "auto" if not hw_cfg.multi_gpu else {"": "cuda"}

        bnb = build_bnb_config(hw_cfg)
        if bnb is not None:
            kw["quantization_config"] = bnb
        else:
            prec = hw_cfg.precision
            kw["torch_dtype"] = (
                torch.bfloat16 if prec == "bf16" else
                torch.float16  if prec == "fp16" else
                torch.float32
            )

        if hw_cfg.flash_attn:
            kw["attn_implementation"] = "flash_attention_2"

    elif device == "mps":
        kw["torch_dtype"] = torch.float32

    logger.info(f"Loading model: {mp}")
    model = AutoModelForCausalLM.from_pretrained(mp, **kw)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {param_count:,} parameters")

    if hw_cfg.mode in ("lora", "qlora"):
        model = _apply_lora(model, hw_cfg, model_cfg)
    elif hw_cfg.mode == "full":
        use_gc = should_use_gradient_checkpointing(hw_cfg, param_count)
        if use_gc and hw_cfg.quantization == "none":
            try:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled.")
            except Exception as e:
                logger.warning(f"Gradient checkpointing failed: {e}")

    if hw_cfg.compile and device == "cuda":
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer, device


def _apply_lora(model, hw_cfg, model_cfg):
    if not HAS_PEFT:
        logger.warning("peft not installed — falling back to full fine-tune.")
        return model

    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    if hw_cfg.mode == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    existing_lora = getattr(model_cfg, "load_existing_lora", None)
    if existing_lora:
        from peft import PeftModel
        logger.info(f"Loading existing LoRA from: {existing_lora}")
        model = PeftModel.from_pretrained(model, existing_lora, is_trainable=True)
        logger.info("Existing LoRA loaded — continuing training.")
        return model

    lc = hw_cfg
    if hasattr(hw_cfg, "lora_cfg"):
        lc_data = hw_cfg.lora_cfg
    else:
        lc_data = hw_cfg

    try:
        from core.config import LoraConfig as LoraCfg
        if hasattr(lc_data, "r"):
            r       = lc_data.r
            alpha   = lc_data.alpha
            dropout = lc_data.dropout
            tgt     = lc_data.target_modules
        else:
            r, alpha, dropout, tgt = 16, 32, 0.05, None
    except Exception:
        r, alpha, dropout, tgt = 16, 32, 0.05, None

    if tgt is None:
        tgt = _auto_detect_lora_targets(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=tgt,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _auto_detect_lora_targets(model) -> list:
    named = [name for name, _ in model.named_modules()]
    candidates = []
    for name in named:
        last = name.split(".")[-1]
        if any(t in last for t in ["q_proj","v_proj","k_proj","o_proj",
                                    "query","value","key","out_proj",
                                    "c_attn","c_proj"]):
            candidates.append(last)
    unique = list(dict.fromkeys(candidates))
    if unique:
        logger.info(f"Auto-detected LoRA targets: {unique[:8]}")
        return unique[:8]
    return ["q_proj","v_proj"]


def export_to_onnx(model_dir: str, output_dir: str, tokenizer, seq_len: int = 512) -> str:
    logger.info("Exporting to ONNX...")
    try:
        from optimum.exporters.onnx import main_export
        onnx_path = os.path.join(output_dir, "onnx")
        os.makedirs(onnx_path, exist_ok=True)
        main_export(model_name_or_path=model_dir, output=onnx_path,
                    task="text-generation", opset=17)
        logger.info(f"ONNX model saved to: {onnx_path}")
        return onnx_path
    except ImportError:
        logger.warning("optimum not installed. Falling back to torch.onnx.export.")
        return _export_onnx_manual(model_dir, output_dir, tokenizer, seq_len)
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return ""


def _export_onnx_manual(model_dir: str, output_dir: str, tokenizer, seq_len: int) -> str:
    if not HAS_TORCH:
        logger.error("PyTorch not available for ONNX export.")
        return ""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        model.eval()

        dummy = tokenizer("Hello world", return_tensors="pt",
                          max_length=seq_len, truncation=True, padding="max_length")
        onnx_path = os.path.join(output_dir, "model.onnx")

        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            onnx_path,
            input_names=["input_ids","attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0:"batch",1:"seq"},
                "attention_mask": {0:"batch",1:"seq"},
                "logits": {0:"batch",1:"seq"},
            },
            opset_version=17,
        )
        logger.info(f"ONNX saved: {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.error(f"Manual ONNX export failed: {e}")
        return ""


def convert_to_gptq(model_dir: str, output_dir: str, bits: int = 4, dataset_id: str = "wikitext") -> str:
    logger.info(f"Converting to GPTQ {bits}-bit...")
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        quant_cfg  = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
        model      = AutoGPTQForCausalLM.from_pretrained(model_dir, quant_cfg)

        ds = load_dataset(dataset_id, "wikitext-2-raw-v1", split="train[:128]")
        examples = [tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                    for text in ds["text"][:128] if text.strip()]

        model.quantize(examples)
        os.makedirs(output_dir, exist_ok=True)
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"GPTQ model saved: {output_dir}")
        return output_dir
    except ImportError:
        logger.error("auto-gptq not installed. Run: pip install auto-gptq")
        return ""
    except Exception as e:
        logger.error(f"GPTQ conversion failed: {e}")
        return ""


def run_chat_test(model_dir: str, tokenizer=None):
    logger.info("Loading model for chat test...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.float16 if HAS_TORCH and torch.cuda.is_available() else torch.float32,
            device_map="auto" if HAS_TORCH and torch.cuda.is_available() else None,
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        print("\n\033[1m\033[96m  ─── Chat Test ─── (type 'exit' to quit)\033[0m\n")
        while True:
            try:
                user_input = input("\033[1m  You:\033[0m ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                prompt = f"### Human:\n{user_input}\n\n### Assistant:\n"
                result = pipe(prompt, max_new_tokens=256, temperature=0.7,
                               top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                response = result[0]["generated_text"][len(prompt):]
                print(f"\033[92m  Yuuki:\033[0m {response.strip()}\n")
            except KeyboardInterrupt:
                break
    except Exception as e:
        logger.error(f"Chat test failed: {e}")


def generate_synthetic_data(model_dir: str, n_samples: int, topics: list = None) -> list:
    logger.info(f"Generating {n_samples} synthetic training samples...")
    if topics is None:
        topics = ["programming", "mathematics", "general knowledge", "creative writing",
                  "science", "history", "coding help", "explanations"]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.float16 if HAS_TORCH and torch.cuda.is_available() else torch.float32,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        samples = []
        import random as rng
        for i in range(n_samples):
            topic = rng.choice(topics)
            seed_prompt = f"Generate a helpful Q&A about {topic}:\nQ:"
            result = pipe(seed_prompt, max_new_tokens=200, temperature=0.8,
                           do_sample=True, pad_token_id=tokenizer.eos_token_id)
            text = result[0]["generated_text"]
            samples.append({"text": text, "source": "synthetic"})
            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i+1}/{n_samples} samples")

        return samples
    except Exception as e:
        logger.error(f"Synthetic data generation failed: {e}")
        return []
