## 1. Project Purpose (Why This Project Exists)

This project is **not** about achieving the highest possible accuracy.
The core purpose is to:

- Analyze the internal structure of the LG **EXAONE** language model
- Systematically **reduce model parameters (compression)**
- Quantitatively evaluate how each structural change affects:
  - **Accuracy**
  - **Precision**
- Clearly explain **which parameters were removed, why they were chosen, and what trade-offs occurred**

This project evaluates the ability to:
> **understand, modify, and justify changes to a large language model**,  
not merely tune hyperparameters.

---

## 2. What Success Looks Like

A successful outcome demonstrates:

- Clear **model variants** created via structural compression
- Explicit documentation of:
  - removed layers / heads / hidden dimensions
- Reproducible experiments with:
  - fixed data
  - fixed training setup
- Quantitative comparison:
  - performance vs. model size / parameter count
- A defensible final choice of a “best compressed model”

High score alone is insufficient without explanation.

---

## 3. What This Project Is NOT

The agent must NOT treat this as:

- A Kaggle-style leaderboard optimization task
- A pure hyperparameter tuning problem
- A black-box fine-tuning exercise
- A full MLOps / production deployment project

Do NOT introduce:
- CI/CD pipelines
- Web APIs
- Database servers
- Distributed training
- Over-engineered infrastructure

Engineering complexity must serve **model understanding**, not replace it.

---

## 4. Core Project Structure (Mental Model)

The project is organized around **model variants**, not experiments.

Original EXAONE
↓
Structural Variant (compression)
↓
Training (same conditions)
↓
Evaluation (accuracy / precision)
↓
Comparison & justification

Each variant answers the question:
> “What happens if we remove *this* part of the model?”

---

## 5. Code Structure Philosophy

### 5.1 Model-Centric Design

- `src/models/base/`
  - Contains the **unaltered EXAONE reference model**
- `src/models/variants/`
  - Each file represents **one compression strategy**
  - Examples:
    - drop_layers
    - prune_heads
    - reduce_hidden_dim

Each variant must:
- Be minimal
- Clearly show **what is removed**
- Be reversible and comparable

---

### 5.2 Configuration Is Evidence

All structural changes must be expressed in **Hydra config files**.

Config files are not just settings — they are **experimental evidence**.

Example intent:

```yaml
compression:
  drop_layers: [8, 9, 10]

This answers:

“Which layers were removed in this experiment?”

Hard-coding compression logic without config exposure is forbidden.
```
---

6. Experiment Execution Rules
	•	All experiments must be runnable via:
	•	make train
	•	All runs must use:
	•	the same dataset
	•	the same training loop
	•	Only one compression factor changes per experiment
	•	No combined compression unless explicitly designed as a final experiment

---

7. MLflow Usage Policy (Mandatory)

MLflow is the single source of truth for results.

Each run must log:

Metrics
	•	accuracy
	•	precision

Structural Parameters
	•	number_of_layers
	•	number_of_heads
	•	hidden_dim
	•	total_parameters
	•	model_size_mb
	•	inference_latency_ms (if available)

Tags
	•	compression_type
	•	variant_name
	•	experiment_stage (baseline / compression / final)

If a result is not logged in MLflow, it does not exist.

---

8. How the Agent Should Help

When assisting, the agent should prioritize:
	1.	Identifying structural redundancy
	2.	Proposing safe compression candidates
	3.	Designing controlled experiments
	4.	Improving explanability of results
	5.	Helping generate:
	•	comparison tables
	•	rationale text
	•	submission-ready summaries

The agent should always ask:

“Does this change help explain why this part of the model can be removed?”

---

9. How the Agent Should NOT Help

The agent must NOT:
	•	Suggest blind hyperparameter sweeps
	•	Optimize for leaderboard score without analysis
	•	Introduce unrelated architectures
	•	Hide compression logic inside training code
	•	Propose changes that reduce explainability

If a suggestion improves performance but reduces interpretability,
it should be flagged as misaligned with project goals.

---

10. Collaboration & Discipline
	•	Small, focused changes only
	•	One variant per PR
	•	Every compression decision must have:
	•	a hypothesis
	•	a measurable outcome
	•	a written justification

---

11. Final Guiding Principle

This project values “understanding a model” more than “beating a score.”
Every line of code should make the model easier to reason about.

The agent’s role is to amplify clarity, not complexity.

End of instructions.