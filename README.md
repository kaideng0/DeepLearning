# Deep Learning Projects

A collection of hands-on deep learning projects, experiments, and reusable implementations. This repository is intended for exploring neural-network architectures, training techniques, and applications across areas such as computer vision, natural language processing, and generative AI.

## Projects

Projects will be listed here as they are added.

| Project | Description | Stack |
| --- | --- | --- |
| [TerraClass](projects/terraclass/) | Satellite land-use classification with a custom CNN and ResNet18 transfer learning. | PyTorch, TorchVision |
| [SupportRouter](projects/support-router/) | Banking-support intent routing with a scratch Transformer, DistilBERT, and confidence-aware human escalation. | PyTorch, Transformers |

## Suggested structure

Each project should be self-contained and include its own README, dependencies, source code, and tests where appropriate.

```text
.
├── projects/
│   └── project-name/
│       ├── README.md
│       ├── requirements.txt
│       ├── notebooks/
│       ├── src/
│       └── tests/
└── README.md
```

Large datasets, trained weights, and generated experiment artifacts should not be committed to Git. Document where to obtain them in the relevant project README instead.

## Getting started

Clone the repository, enter a project directory, and create an isolated Python environment:

```bash
git clone <repository-url>
cd DeepLearning/projects/<project-name>
python -m venv .venv
```

Activate the environment and install that project's dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Refer to each project's README for its data setup, training, evaluation, and inference instructions.

## Contributing

Keep projects focused and reproducible. When adding a project:

1. Place it under `projects/`.
2. Include setup and usage instructions in a project-level README.
3. Pin or constrain dependencies.
4. Avoid committing datasets, credentials, checkpoints, or generated outputs.
5. Add tests for reusable code when practical.

## License

No license has been selected yet. Until one is added, all rights are reserved.
