# Data Directory

This directory is for storing large datasets (e.g., C4, WikiText) and model checkpoints.
**Git ignores all files in this directory except this README.**

## Recommended Structure
- `data/raw/`: Original datasets (immutable)
- `data/processed/`: Tokenized or preprocessed data
- `data/checkpoints/`: Model weights (fine-tuned or compressed)
