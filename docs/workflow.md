# Workflow

This document describes the basic workflow of the particle detection pipeline.

---

# 1. Generate Synthetic Dataset

Generate synthetic images and segmentation masks:

```bash
python data_generation/generate.py
```

---

# 2. Convert Dataset to NumPy Format

Convert the generated dataset into `.npy` format for efficient training access:

```bash
python tools/convert_to_npy.py
```

---

# 3. Start Model Training

Run training inside the CUDA-enabled Docker container:

```bash
python training/train.py
```

Training parameters can be configured in:

```txt
configs/training.yaml
```

---

# 4. Export ONNX Model

Export the best trained model checkpoint to ONNX format:

```bash
python export/export_onnx.py
```

---

# Pipeline Overview

```txt
Data Generation
      ↓
Dataset Conversion
      ↓
Model Training
      ↓
ONNX Export
      ↓
Inference
```