# Model Variants

This directory contains compressed variants of the EXAONE model.

## Structure
Each file should represent a specific structural compression strategy (e.g., `drop_layers.py`, `prune_heads.py`).

## Requirements
- Each variant must be reproducible via config.
- Clearly document what parameters are removed.

## Why These Variants?
- drop_layers:
  - Hypothesis: upper layers contribute less to task-specific performance
- prune_heads:
  - Hypothesis: redundant attention heads exist
- reduce_hidden:
  - Hypothesis: representation over-parameterization