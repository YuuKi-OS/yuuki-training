from __future__ import annotations
import logging, time, threading, shutil, os, json
from typing import Optional

logger = logging.getLogger("train_yuuki.callbacks")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from transformers import TrainerCallback

C  = "\033[96m"
G  = "\033[92m"
Y  = "\033[93m"
R  = "\033[91m"
B  = "\033[1m"
RS = "\033[0m"


def _hs(b: float) -> str:
    if b < 1024:   return f"{b:.0f} B"
    if b < 1<<20:  return f"{b/1024:.1f} KB"
    if b < 1<<30:  return f"{b/(1<<20):.1f} MB"
    return f"{b/(1<<30):.2f} GB"


class DockerProgressCallback(TrainerCallback):
    def __init__(self, log_queue=None):
        self._t0    = None
        self._s0    = None
        self._total = None
        self._log_q = log_queue

    def on_train_begin(self, args, state, control, **kw):
        self._total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        self._s0    = state.global_step
        self._t0    = time.time()
        print()

    def on_step_end(self, args, state, control, **kw):
        step    = state.global_step
        total   = self._total or 0
        loss    = next((e["loss"] for e in reversed(state.log_history) if "loss" in e), None)
        elapsed = time.time() - (self._t0 or time.time())
        done    = step - (self._s0 or 0)

        eta = ""
        if done > 0 and total > 0:
            s = elapsed / done * (total - step)
            h, rem = divmod(int(s), 3600)
            m, sc  = divmod(rem, 60)
            eta = f"  ETA {h:02d}:{m:02d}:{sc:02d}"

        W = 38
        if total > 0:
            fr  = step / total
            fi  = int(W * fr)
            bar = ":" * fi + " " * (W - fi)
            pct = f"{fr*100:5.1f}%"
        else:
            fi  = step % W
            bar = " " * fi + ":::" + " " * max(0, W - fi - 3)
            pct = f"step {step}"

        gpu_str = ""
        if HAS_TORCH and torch.cuda.is_available():
            mem_used = torch.cuda.memory_reserved(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory
            gpu_str = f"  GPU {_hs(mem_used)}/{_hs(mem_total)}"

        ls = f"  loss {loss:.4f}" if loss else ""
        line = f"\r  {C}Training{RS} [{G}{bar}{RS}] {pct}{ls}{eta}{gpu_str}   "
        print(line, end="", flush=True)

        if self._log_q is not None:
            try:
                self._log_q.put_nowait({
                    "step": step, "total": total,
                    "loss": loss, "eta": eta.strip(),
                    "pct": round(step/total*100, 1) if total > 0 else 0,
                })
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kw):
        print(f"\r  {C}Training{RS} [{G}{'='*38}{RS}] {G}Done ✓{RS}                                         ")


class ResourceMonitorCallback(TrainerCallback):
    def __init__(self, check_every: int = 50, mem_threshold: float = 0.12,
                 cpu_threshold: int = 95):
        self.check_every     = check_every
        self.mem_threshold   = mem_threshold
        self.cpu_threshold   = cpu_threshold

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.check_every != 0 or not HAS_PSUTIL:
            return control
        try:
            vm  = psutil.virtual_memory()
            av  = vm.available / vm.total
            cpu = psutil.cpu_percent(interval=None)
            if av < self.mem_threshold or cpu >= self.cpu_threshold:
                logger.warning(
                    f"Resource pressure at step {state.global_step}: "
                    f"mem={av:.2f} cpu={cpu:.0f}% — forcing checkpoint save."
                )
                control.should_save = True
        except (PermissionError, OSError):
            pass
        return control


class AdaptiveCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, limit: int, total_steps: Optional[int] = None):
        self.output_dir  = output_dir
        self.limit       = limit
        self.total_steps = total_steps

    def _get_save_interval(self, state) -> int:
        if not self.total_steps or self.total_steps <= 0:
            return 100
        step = state.global_step
        progress = step / self.total_steps
        if progress < 0.1:
            return 25
        if progress < 0.3:
            return 50
        if progress < 0.7:
            return 100
        return 200

    def on_save(self, args, state, control, **kw):
        if not os.path.exists(self.output_dir):
            return control
        ckpts = sorted([
            (int(d.split("-")[-1]), os.path.join(self.output_dir, d))
            for d in os.listdir(self.output_dir)
            if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
        ], reverse=True)
        for _, path in ckpts[self.limit:]:
            try:
                shutil.rmtree(path)
                logger.info(f"Removed old checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Could not remove {path}: {e}")
        return control


class EvalPerplexityCallback(TrainerCallback):
    def __init__(self, eval_every: int = 200, log_queue=None):
        self.eval_every = eval_every
        self._log_q     = log_queue

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % self.eval_every == 0 and state.global_step > 0:
            if state.log_history:
                last = next((e for e in reversed(state.log_history) if "eval_loss" in e), None)
                if last:
                    import math
                    ppl = math.exp(min(last["eval_loss"], 20))
                    logger.info(f"Step {state.global_step} — eval perplexity: {ppl:.2f}")
                    if self._log_q:
                        try:
                            self._log_q.put_nowait({"perplexity": ppl, "step": state.global_step})
                        except Exception:
                            pass
        return control


class StopFlagCallback(TrainerCallback):
    def __init__(self, flag_ref: list):
        self._flag = flag_ref

    def on_step_end(self, args, state, control, **kw):
        if self._flag and self._flag[0]:
            control.should_training_stop = True
        return control


class WebhookNotifier(TrainerCallback):
    def __init__(self, cfg):
        self.cfg        = cfg
        self._start_t   = None
        self._last_ckpt = None

    def _send(self, message: str):
        if not self.cfg.enabled or not self.cfg.url:
            return
        try:
            import urllib.request
            platform = self.cfg.platform.lower()
            if platform == "discord":
                payload = json.dumps({"content": f"🤖 **Yuuki Trainer**\n{message}"}).encode()
                headers = {"Content-Type": "application/json"}
            elif platform == "slack":
                payload = json.dumps({"text": f"🤖 Yuuki Trainer\n{message}"}).encode()
                headers = {"Content-Type": "application/json"}
            elif platform == "telegram":
                payload = json.dumps({"text": f"🤖 Yuuki Trainer\n{message}"}).encode()
                headers = {"Content-Type": "application/json"}
            else:
                payload = json.dumps({"message": message}).encode()
                headers = {"Content-Type": "application/json"}

            req = urllib.request.Request(self.cfg.url, data=payload, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")

    def on_train_begin(self, args, state, control, **kw):
        self._start_t = time.time()
        self._send(f"Training started ▶\nSteps: {state.max_steps or '?'}")

    def on_save(self, args, state, control, **kw):
        if self.cfg.on_checkpoint:
            self._send(f"Checkpoint saved 💾 step {state.global_step}")

    def on_train_end(self, args, state, control, **kw):
        if not self.cfg.on_finish:
            return
        elapsed = time.time() - (self._start_t or time.time())
        h, r = divmod(int(elapsed), 3600)
        m, s = divmod(r, 60)
        loss = next((e["loss"] for e in reversed(state.log_history) if "loss" in e), None)
        self._send(
            f"Training complete ✅\n"
            f"Steps: {state.global_step}\n"
            f"Time: {h}h {m:02d}m {s:02d}s\n"
            f"Final loss: {loss:.4f}" if loss else "Training complete ✅"
        )

    def notify_crash(self, error: str):
        if self.cfg.on_crash:
            self._send(f"Training crashed ❌\n```{error[:500]}```")

    def notify_upload(self, repo: str):
        self._send(f"Model uploaded to HuggingFace 🚀\nhttps://huggingface.co/{repo}")


def build_callbacks(cfg, output_dir: str, total_steps: Optional[int],
                    stop_flag: list, log_queue=None) -> list:
    callbacks = [
        DockerProgressCallback(log_queue=log_queue),
        ResourceMonitorCallback(),
        AdaptiveCheckpointCallback(output_dir, cfg.hp.save_limit, total_steps),
        StopFlagCallback(stop_flag),
        EvalPerplexityCallback(cfg.hp.eval_steps, log_queue=log_queue),
    ]
    if cfg.webhook.enabled:
        callbacks.append(WebhookNotifier(cfg.webhook))
    return callbacks
