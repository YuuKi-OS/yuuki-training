from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    source: str = "hf_id"
    model_path: str = "OpceanAI/Yuuki-NxG"
    gguf_path: Optional[str] = None
    model_is_hf: bool = True
    load_existing_lora: Optional[str] = None


@dataclass
class HardwareConfig:
    mode: str = "lora"
    use_gpu: bool = True
    precision: str = "bf16"
    quantization: str = "none"
    flash_attn: bool = False
    compile: bool = False
    multi_gpu: bool = False


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    merge: bool = False


@dataclass
class DatasetEntry:
    type: str = "hf"
    id: Optional[str] = None
    path: Optional[str] = None
    config: Optional[str] = None
    split: str = "train"
    label: str = ""
    format: str = "auto"


@dataclass
class DatasetConfig:
    datasets: List[DatasetEntry] = field(default_factory=list)
    buffer: int = 1000
    extend_tokenizer: bool = False
    preview_before_train: bool = True
    val_split_ratio: float = 0.05


@dataclass
class HyperParams:
    output_dir: str = "./yuuki_output"
    max_length: int = 512
    batch_size: int = 1
    grad_accum: int = 8
    epochs: int = 3
    max_steps: int = -1
    lr: float = 2e-5
    warmup: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    save_steps: int = 100
    save_limit: int = 3
    eval_steps: int = 200
    seed: int = 42
    smart_gradient_checkpointing: bool = True
    adaptive_checkpoint_schedule: bool = True


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "yuuki-nxg"
    wandb_api_key: str = ""
    use_tensorboard: bool = False
    tensorboard_dir: str = "./runs"


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    platform: str = "discord"
    on_finish: bool = True
    on_crash: bool = True
    on_checkpoint: bool = False


@dataclass
class PostTrainingConfig:
    chat_test: bool = True
    convert_gguf: bool = False
    gguf_quant: str = "Q4_K_M"
    gguf_target: str = "ollama"
    convert_onnx: bool = False
    convert_gptq: bool = False
    gptq_bits: int = 4
    upload_hf: bool = False
    hf_repo: str = ""
    hf_token: str = ""
    hf_private: bool = False
    synthetic_data: bool = False
    synthetic_samples: int = 1000


@dataclass
class DPOConfig:
    enabled: bool = False
    beta: float = 0.1
    max_prompt_length: int = 256
    max_length: int = 512
    dataset_id: str = ""


@dataclass
class OptimizationConfig:
    use_optuna: bool = False
    optuna_trials: int = 10
    optuna_metric: str = "eval_loss"
    search_lr: bool = True
    search_batch: bool = False
    search_lora_r: bool = False


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hp: HyperParams = field(default_factory=HyperParams)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    post: PostTrainingConfig = field(default_factory=PostTrainingConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    version: str = "2.0"

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.hp.output_dir, "training_config.json")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        cfg.model   = ModelConfig(**data.get("model", {}))
        cfg.hw      = HardwareConfig(**data.get("hw", {}))
        cfg.lora    = LoraConfig(**data.get("lora", {}))
        raw_ds      = data.get("dataset", {})
        entries     = [DatasetEntry(**e) for e in raw_ds.pop("datasets", [])]
        cfg.dataset = DatasetConfig(datasets=entries, **raw_ds)
        cfg.hp      = HyperParams(**data.get("hp", {}))
        cfg.logging = LoggingConfig(**data.get("logging", {}))
        cfg.webhook = WebhookConfig(**data.get("webhook", {}))
        cfg.post    = PostTrainingConfig(**data.get("post", {}))
        cfg.dpo     = DPOConfig(**data.get("dpo", {}))
        cfg.optim   = OptimizationConfig(**data.get("optim", {}))
        return cfg

    def validate(self) -> List[str]:
        errors = []
        if not self.model.model_path and not self.model.gguf_path:
            errors.append("No model specified.")
        if self.hw.mode == "qlora" and self.hw.quantization not in ("int4", "none"):
            errors.append("QLoRA requires int4 quantization.")
        if self.hp.batch_size < 1:
            errors.append("Batch size must be >= 1.")
        if not self.dataset.datasets:
            errors.append("No datasets configured.")
        if self.dpo.enabled and not self.dpo.dataset_id:
            errors.append("DPO enabled but no dataset specified.")
        return errors

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model.model_path or self.model.gguf_path,
            "Mode": self.hw.mode.upper(),
            "Device": f"GPU ({self.hw.precision}/{self.hw.quantization})" if self.hw.use_gpu else "CPU",
            "Multi-GPU": "yes" if self.hw.multi_gpu else "no",
            "Flash Attention": "yes" if self.hw.flash_attn else "no",
            "LoRA r/alpha": f"{self.lora.r}/{self.lora.alpha}" if self.hw.mode in ("lora","qlora") else "N/A",
            "Effective batch": self.hp.batch_size * self.hp.grad_accum,
            "Epochs": self.hp.epochs,
            "Max steps": "unlimited" if self.hp.max_steps == -1 else self.hp.max_steps,
            "LR": self.hp.lr,
            "Scheduler": self.hp.lr_scheduler,
            "Datasets": ", ".join(d.label for d in self.dataset.datasets),
            "DPO": "yes" if self.dpo.enabled else "no",
            "W&B": "yes" if self.logging.use_wandb else "no",
            "Post GGUF": self.post.gguf_quant if self.post.convert_gguf else "no",
            "Post ONNX": "yes" if self.post.convert_onnx else "no",
            "Post GPTQ": f"{self.post.gptq_bits}bit" if self.post.convert_gptq else "no",
            "Upload HF": self.post.hf_repo if self.post.upload_hf else "no",
            "Webhook": self.webhook.platform if self.webhook.enabled else "no",
        }
