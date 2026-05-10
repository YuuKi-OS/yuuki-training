#!/usr/bin/env python3
"""
train_yuuki_refined.py

Refined training script for Yuuki (mobile-first).

Features:
- Detects whether CUDA is available and adjusts some args.
- Skips tokenization if dataset already tokenized (caches arrows).
- ResourceMonitorCallback: autosaves when RAM or CPU thresholds hit.
- Autosave every SAVE_STEPS (default 500) and keeps only SAVE_TOTAL_LIMIT checkpoints.
- Pretty progress bar (tqdm) and lightweight logging.
- Graceful SIGINT handler that forces a manual save before exiting.

Designed for Termux / mobile but also works on desktop (it will auto-detect device).

Run: python train_yuuki_refined.py

Note: optional dependencies: psutil and tqdm. Install if missing: pip install psutil tqdm
"""

import os
import math
import signal
import logging
import sys
from pathlib import Path

# Optional imports
try:
    import psutil
except Exception:
    psutil = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# -----------------------
# Config (tweak for your device)
# -----------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
DATASET_ID = os.environ.get("DATASET_ID", "bigcode/the-stack-smol-xl")
SPLIT = os.environ.get("SPLIT", "train")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./yuuki_model")
TOKENIZED_CACHE_DIR = os.environ.get("TOKENIZED_CACHE_DIR", os.path.expanduser("~/yuuki/tokenized_cache"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))  # v0.1 - 4x más rápido que 512
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION = int(os.environ.get("GRADIENT_ACCUMULATION", "4"))  # Reducido de 8 a 4 para pasos más rápidos

# Effective batch will be printed at start: BATCH_SIZE * GRADIENT_ACCUMULATION

# Autosave frequent for mobile small steps
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))  # Guarda checkpoint cada 50 pasos (muy frecuente para móvil)
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "5"))  # Mantiene solo los últimos 5 checkpoints (borra automáticamente los más antiguos)
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))  # Log cada 10 pasos

# Resources thresholds
CHECK_RESOURCES_EVERY_N_STEPS = int(os.environ.get("CHECK_RESOURCES_EVERY_N_STEPS", "50"))
MEMORY_THRESHOLD = float(os.environ.get("MEMORY_THRESHOLD", "0.12"))  # fraction available
CPU_THRESHOLD = int(os.environ.get("CPU_THRESHOLD", "95"))

# Map batch for tokenization (reduce if memory issues)
MAP_BATCH_SIZE = int(os.environ.get("MAP_BATCH_SIZE", "128"))

# Safety limits (don't change unless you know what you do)
MIN_FREE_RAM_MB = 80  # try to keep at least this free RAM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_yuuki")

# -----------------------
# Utility helpers
# -----------------------

def has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def human_size(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"

def get_last_checkpoint(output_dir):
    """Find the most recent checkpoint directory in output_dir."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                step_num = int(item.split("-")[-1])
                checkpoints.append((step_num, item_path))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Return the checkpoint with the highest step number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

def cleanup_old_checkpoints(output_dir, max_checkpoints=5):
    """Ensure we never have more than max_checkpoints. Delete oldest ones if exceeded."""
    if not os.path.exists(output_dir):
        return
    
    checkpoints = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                step_num = int(item.split("-")[-1])
                checkpoints.append((step_num, item_path))
            except ValueError:
                continue
    
    # Sort by step number (newest first)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    # If we have more than max_checkpoints, delete the oldest ones
    if len(checkpoints) > max_checkpoints:
        to_delete = checkpoints[max_checkpoints:]
        for step_num, checkpoint_path in to_delete:
            logger.info(f"Eliminando checkpoint antiguo: {checkpoint_path}")
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
            except Exception as e:
                logger.warning(f"No se pudo eliminar {checkpoint_path}: {e}")

# -----------------------
# Callbacks
# -----------------------

class ResourceMonitorCallback(TrainerCallback):
    """Checks memory and CPU every N steps and requests a checkpoint save when thresholds are exceeded."""

    def __init__(self, check_every_n=50, mem_threshold=0.12, cpu_threshold=95):
        self.check_every_n = check_every_n
        self.mem_threshold = mem_threshold
        self.cpu_threshold = cpu_threshold
        self._step = 0
        self.psutil = psutil

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.check_every_n != 0:
            return control
        if self.psutil is None:
            return control
        
        try:
            vm = self.psutil.virtual_memory()
            avail_frac = vm.available / vm.total if vm.total else 1.0
            cpu = int(self.psutil.cpu_percent(interval=None))
            logger.debug(f"Resource check @ step {state.global_step}: avail_frac={avail_frac:.2f}, cpu={cpu}%")
            if avail_frac < self.mem_threshold or cpu >= self.cpu_threshold:
                logger.info(f"Resource threshold exceeded (mem={avail_frac:.2f}, cpu={cpu}%). Requesting save.")
                control.should_save = True
                control.should_log = True
        except (PermissionError, OSError) as e:
            # Termux may not have permissions to access /proc/stat
            logger.debug(f"Cannot check resources (permission denied): {e}")
            pass
        
        return control

class TqdmProgressCallback(TrainerCallback):
    """Simple TQDM progress visualizer that prints loss when available.
    
    Falls back to basic logging if tqdm is not installed.
    """

    def __init__(self, total_steps=None):
        self.total_steps = total_steps
        self.pbar = None
        self._last_log_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if tqdm is None:
            logger.info("tqdm not available — using default logs")
            return
        total = int(state.max_steps) if state.max_steps is not None and state.max_steps > 0 else self.total_steps
        self.pbar = tqdm(total=total, desc="Training", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            # advance one step for each global step change
            self.pbar.n = int(state.global_step)
            # show approximate ETA and a minimal loss if logged
            last = None
            if hasattr(state, 'log_history') and state.log_history:
                for e in reversed(state.log_history):
                    if 'loss' in e:
                        last = e['loss']
                        break
            self.pbar.set_postfix({"loss": f"{last:.4f}" if last is not None else "-"})
            self.pbar.refresh()

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

class CheckpointCleanupCallback(TrainerCallback):
    """Cleans up old checkpoints after each save to maintain max limit."""

    def __init__(self, output_dir, max_checkpoints=5):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        cleanup_old_checkpoints(self.output_dir, max_checkpoints=self.max_checkpoints)
        return control

# -----------------------
# Signal handler for graceful save
# -----------------------

_save_requested = False

def _signal_handler(sig, frame):
    global _save_requested
    logger.warning("SIGINT received — will request a graceful save and stop after current step.")
    _save_requested = True

signal.signal(signal.SIGINT, _signal_handler)

# -----------------------
# Main
# -----------------------

def main():
    device = "cuda" if has_cuda() else "cpu"
    logger.info(f"Device detected: {device}")
    
    if device == "cpu":
        logger.warning("⚠️  Entrenando en CPU - esto será MUY LENTO")
        logger.warning("⚠️  Considera reducir MAX_LENGTH o usar un modelo más pequeño")
        logger.warning("⚠️  Tiempo estimado: ~5 minutos por paso")

    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
    logger.info(f"Per-device batch_size={BATCH_SIZE}, gradient_accumulation_steps={GRADIENT_ACCUMULATION}, effective batch={effective_batch}")

    # Create tokenized cache directory if it doesn't exist
    os.makedirs(TOKENIZED_CACHE_DIR, exist_ok=True)
    logger.info(f"Cache de tokenización: {TOKENIZED_CACHE_DIR}")

    # Check for existing checkpoint to resume from
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        logger.info(f"¡Checkpoint encontrado! Reanudando desde: {last_checkpoint}")
    else:
        logger.info("No se encontró checkpoint previo. Iniciando entrenamiento desde cero.")
    
    # Clean up old checkpoints if there are more than 5
    cleanup_old_checkpoints(OUTPUT_DIR, max_checkpoints=SAVE_TOTAL_LIMIT)

    # Load dataset (cached if present)
    logger.info("Cargando dataset (puede tardar)...")
    dataset = load_dataset(DATASET_ID, split=SPLIT)

    # If already tokenized (has input_ids column), skip tokenization
    tokenized_already = 'input_ids' in dataset.column_names

    # Tokenizer
    logger.info("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(batch):
        key = "code" if "code" in batch else ("content" if "content" in batch else list(batch.keys())[0])
        toks = tokenizer(batch[key], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        toks["labels"] = toks["input_ids"].copy()
        return toks

    if not tokenized_already:
        logger.info("Tokenizando dataset (esto puede tardar; usa batched=True)...")
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            remove_columns=[c for c in dataset.column_names],
            cache_file_name=os.path.join(TOKENIZED_CACHE_DIR, "tokenized_dataset.arrow"),
        )
    else:
        logger.info("Dataset ya tokenizado — saltando tokenización.")

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    logger.info("Cargando modelo...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        logger.debug("Gradient checkpointing not available for this model")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Compute steps
    try:
        total_examples = len(dataset)
        steps_per_epoch = math.ceil(total_examples / (BATCH_SIZE * GRADIENT_ACCUMULATION))
        max_steps = steps_per_epoch * EPOCHS
        logger.info(f"Total examples: {total_examples}, steps/epoch: {steps_per_epoch}, max_steps: {max_steps}")
    except Exception:
        steps_per_epoch = None
        max_steps = None

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=(device == "cuda"),
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        dataloader_num_workers=0,  # Cambiado a 0 para evitar overhead en móvil
        dataloader_pin_memory=False,  # Desactivado para CPU
        remove_unused_columns=False,
        report_to=[],
        resume_from_checkpoint=last_checkpoint,  # Auto-resume from last checkpoint
        # Optimizaciones adicionales para móvil
        gradient_checkpointing=True,
        optim="adamw_torch",  # Optimizador más rápido
        max_grad_norm=1.0,
    )

    # Callbacks
    resource_cb = ResourceMonitorCallback(
        check_every_n=CHECK_RESOURCES_EVERY_N_STEPS,
        mem_threshold=MEMORY_THRESHOLD,
        cpu_threshold=CPU_THRESHOLD,
    )
    progress_cb = TqdmProgressCallback(total_steps=max_steps)
    cleanup_cb = CheckpointCleanupCallback(output_dir=OUTPUT_DIR, max_checkpoints=SAVE_TOTAL_LIMIT)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        callbacks=[resource_cb, progress_cb, cleanup_cb],
    )

    # Wrap training loop to support graceful save on SIGINT or resource request
    if last_checkpoint:
        logger.info(f"Inicio de entrenamiento — {EPOCHS} epochs configuradas (reanudando desde checkpoint).")
    else:
        logger.info(f"Inicio de entrenamiento — {EPOCHS} epochs configuradas.")

    # Helper to do a manual save
    def manual_save(tag: str = None):
        dest = OUTPUT_DIR
        if tag:
            dest = os.path.join(OUTPUT_DIR, f"manual-{tag}")
        logger.info(f"Guardando modelo manualmente en: {dest}")
        trainer.save_model(dest)
        tokenizer.save_pretrained(dest)

    # Start training with periodic checks
    try:
        # We can't easily interrupt trainer.train() internally, so we rely on callbacks and SIGINT
        # Pass resume_from_checkpoint to continue from where we left off
        trainer.train(resume_from_checkpoint=last_checkpoint)
    except KeyboardInterrupt:
        logger.warning("Interrupción por el usuario detectada. Guardando checkpoint...")
        try:
            manual_save(tag=f"step{trainer.state.global_step}")
        except Exception as e:
            logger.exception("Error al guardar el checkpoint durante interrupción: %s", e)
        logger.info("Guardado finalizado. Saliendo.")
        sys.exit(0)

    # Final save
    logger.info("Entrenamiento terminado. Guardando modelo final...")
    manual_save(tag="final")
    logger.info("Entrenamiento completado. Modelo guardado en: %s", OUTPUT_DIR)

if __name__ == "__main__":
    main()

