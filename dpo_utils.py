from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger("train_yuuki.dpo")


def run_dpo_training(model, tokenizer, cfg, output_dir: str, stop_flag: list):
    logger.info("Starting DPO training...")
    try:
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    except ImportError:
        logger.error("trl not installed. Run: pip install trl")
        return None

    from core.dataset_utils import build_dpo_dataset

    dpo_ds = build_dpo_dataset(cfg.dpo.dataset_id, tokenizer, cfg)

    dpo_config = TRLDPOConfig(
        output_dir=output_dir,
        beta=cfg.dpo.beta,
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
        per_device_train_batch_size=cfg.hp.batch_size,
        gradient_accumulation_steps=cfg.hp.grad_accum,
        learning_rate=cfg.hp.lr,
        num_train_epochs=cfg.hp.epochs,
        save_steps=cfg.hp.save_steps,
        logging_steps=10,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_ds,
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("DPO training interrupted.")

    return trainer


def run_optuna_search(build_model_fn, build_data_fn, cfg, n_trials: int = 10) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        return {}

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    import math

    logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        search_lr    = cfg.optim.search_lr
        search_batch = cfg.optim.search_batch
        search_lora  = cfg.optim.search_lora_r

        trial_lr    = trial.suggest_float("lr", 1e-6, 1e-3, log=True) if search_lr else cfg.hp.lr
        trial_batch = trial.suggest_categorical("batch", [1, 2, 4]) if search_batch else cfg.hp.batch_size
        trial_r     = trial.suggest_categorical("lora_r", [8, 16, 32, 64]) if search_lora else getattr(cfg.lora, "r", 16)

        trial_cfg = cfg
        trial_cfg.hp.lr = trial_lr
        trial_cfg.hp.batch_size = trial_batch
        if hasattr(trial_cfg, "lora"):
            trial_cfg.lora.r = trial_r

        try:
            model, tokenizer, device = build_model_fn(trial_cfg)
            train_ds = build_data_fn(trial_cfg, tokenizer)

            args = TrainingArguments(
                output_dir=f"{cfg.hp.output_dir}/optuna_trial_{trial.number}",
                max_steps=50,
                per_device_train_batch_size=trial_batch,
                gradient_accumulation_steps=cfg.hp.grad_accum,
                learning_rate=trial_lr,
                fp16=(device == "cuda" and cfg.hw.precision == "fp16"),
                bf16=(device == "cuda" and cfg.hw.precision == "bf16"),
                report_to=[],
                logging_steps=10,
                remove_unused_columns=False,
                dataloader_num_workers=0,
            )

            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer  = Trainer(model=model, args=args, data_collator=collator,
                               train_dataset=train_ds)
            result = trainer.train()

            eval_loss = result.training_loss
            logger.info(f"Trial {trial.number}: lr={trial_lr:.2e} batch={trial_batch} r={trial_r} loss={eval_loss:.4f}")
            return eval_loss

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Best hyperparameters: {best}")
    logger.info(f"Best loss: {study.best_value:.4f}")

    return best


def apply_optuna_results(cfg, best_params: dict):
    if "lr" in best_params:
        cfg.hp.lr = best_params["lr"]
    if "batch" in best_params:
        cfg.hp.batch_size = best_params["batch"]
    if "lora_r" in best_params and hasattr(cfg, "lora"):
        cfg.lora.r = best_params["lora_r"]
        cfg.lora.alpha = best_params["lora_r"] * 2
    return cfg
