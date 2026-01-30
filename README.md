<div align="center">

# Yuuki Training Code

**Official training pipeline for Yuuki, an experimental small-scale language model for source code generation.**

[![Model](https://img.shields.io/badge/HuggingFace-Yuuki--82M-yellow)](https://huggingface.co/OpceanAI/Yuuki-82M)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)

</div>

---

## Abstract

This repository contains the official training implementation for **Yuuki**, a compact causal language model optimized for source code understanding and generation. The system is designed with an emphasis on simplicity, reproducibility, and accessibility across heterogeneous computing environments, including CPU-only systems, cloud notebooks (Colab, Kaggle), and resource-constrained platforms such as Termux on mobile devices.

---

## Model Specification

| Attribute | Description |
|-----------|-------------|
| **Architecture** | GPT-style autoregressive transformer |
| **Base Model** | `distilgpt2` |
| **Domain** | Source code (multi-language) |
| **Training Corpus** | `bigcode/the-stack-smol-xl` |
| **Parameter Count** | ~82M |
| **Design Principles** | Minimal dependencies, transparent implementation, full reproducibility |

---

## Repository Structure

### Included Components

| File | Description |
|------|-------------|
| `train_yuuki.py` | Complete, self-contained training script |
| `LICENSE` | Apache 2.0 License |

### Excluded Artifacts

The following components are intentionally omitted to maintain repository portability and encourage local reproducibility:

- Pre-trained model weights and checkpoints
- Tokenized datasets and Arrow cache files
- Training logs and metrics
- Experimental or proprietary scripts
- Auxiliary datasets from subsequent experiments

All artifacts should be generated locally by executing the provided training script.

---

## Configuration Parameters

Training behavior is controlled exclusively through environment variables, enabling seamless adaptation across diverse execution environments.

### Default Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `MODEL_NAME` | `distilgpt2` | Pre-trained model identifier for initialization |
| `DATASET_ID` | `bigcode/the-stack-smol-xl` | HuggingFace dataset identifier |
| `SPLIT` | `train` | Dataset partition for training |
| `OUTPUT_DIR` | `./yuuki_model` | Output directory for model artifacts |
| `TOKENIZED_CACHE_DIR` | `./yuuki_model/tokenized_cache` | Cache location for tokenized sequences |
| `MAX_LENGTH` | `256` | Maximum input sequence length |
| `EPOCHS` | `2` | Number of training iterations |
| `BATCH_SIZE` | `1` | Samples per gradient update |

### Implementation Notes

- **Sequence Length (`MAX_LENGTH=256`)**: Selected to optimize memory utilization and training throughput on constrained hardware.
- **Batch Size (`BATCH_SIZE=1`)**: Configured for compatibility with low-memory execution environments.
- **Tokenization Caching**: Optional but recommended for iterative training workflows.

---

## Execution

### Standard Invocation

```bash
python train_yuuki.py
```

### Custom Configuration Example

```bash
MODEL_NAME=distilgpt2 \
MAX_LENGTH=256 \
EPOCHS=3 \
BATCH_SIZE=2 \
python train_yuuki.py
```

The training script performs automatic hardware detection and configures CUDA acceleration when available.

---

## Design Rationale

Yuuki is not intended to compete with large-scale foundation models. The project objectives are:

| Principle | Description |
|-----------|-------------|
| **Interpretability** | Prioritizes readable, maintainable code over abstraction layers |
| **Accessibility** | Executable without specialized hardware infrastructure |
| **Transparency** | No hidden procedures or undocumented dependencies |
| **Educational Utility** | Serves as a reference implementation for language model training |

---

## Pre-trained Model

The model trained using this pipeline is publicly available:

<div align="center">

**[Yuuki-82M on HuggingFace](https://huggingface.co/OpceanAI/Yuuki-82M)**

</div>

---

## Limitations and Disclaimer

This software is provided for research and educational purposes. The model may produce:

- Syntactically or semantically incorrect code
- Incomplete or truncated outputs
- Potentially unsafe or nonsensical suggestions

**This system is not suitable for production deployment.** Users assume full responsibility for any application of the generated outputs.

---

## License

This project is distributed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete terms.

Under this license, you are permitted to:

- Use, copy, and distribute the software
- Modify and create derivative works
- Use for commercial and non-commercial purposes

Subject to the conditions of attribution and license preservation as specified in the Apache 2.0 terms.

---

## Contact

For inquiries, collaboration proposals, or technical discussions regarding Yuuki, please submit an Issue or initiate a Discussion in this repository.

---

<div align="center">

**Developed by [OpceanAI](https://huggingface.co/OpceanAI)**

</div>

