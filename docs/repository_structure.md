## Repository Structure

```text
particle-detection/
│
├── assets/             # README images and visualizations
├── configs/            # YAML configuration files
├── data/               # datasets and inference outputs
│   ├── generated/          # generated synthetic image and mask samples
│   ├── train_data/         # training input data (.npy)
│   ├── val_data/           # validation and inference input images
│   ├── inference_results/  # batch inference output results
│   └── inference_previews/ # saved outputs from inference_viewer
│
├── data_generation/    # synthetic particle data generation
├── export/             # ONNX export scripts
├── inference/          # ONNX inference pipeline
├── models/             # exported models
├── training/           # model training pipeline
├── tools/              # utility and debugging tools
├── checkpoints/        # training checkpoints
├── logs/               # training logs and CSV metrics
│
├── Dockerfile
├── requirements.txt
└── README.md
```