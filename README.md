# DSA4213_Group19

## Overview
The repository centralizes all data-related code for the project:
- Download and store raw datasets
- Clean, normalize, and export processed datasets
- Produce/consume embeddings for retrieval (optional)
- Provide example notebooks and scripts to run RAG / QA experiments

## Requirements
A minimal set of Python packages is listed in `requirement.txt` at the repository root. 

## Installation (Windows / PowerShell)
Open PowerShell in the entride folder:

1. Create and activate a virtual environment
```powershell
cd "to your local path"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
# if file is named requirement.txt at repo root
pip install -r ..\requirement.txt
```

## Get the PubMedQA dataset
This project expects the labeled PubMedQA split hosted on Hugging Face:

Dataset viewer & link:
https://huggingface.co/datasets/qiaojin/PubMedQA/viewer/pqa_labeled?views%5B%5D=pqa_labeled

Recommended method (python, using datasets):
```python
from datasets import load_dataset

ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
# save to data/raw
ds["train"].to_csv("data/raw/PubMedQA_pqa_labeled_train.csv", index=False)
ds["validation"].to_csv("data/raw/PubMedQA_pqa_labeled_valid.csv", index=False)
ds["test"].to_csv("data/raw/PubMedQA_pqa_labeled_test.csv", index=False)
```

Alternatively download via the Hugging Face web UI or huggingface_hub.

## Configuration & environment variables
Keep secrets out of VCS (use .env). Typical variables:
- AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
- HF_TOKEN (if accessing private Hugging Face resources)
Use python-dotenv or OS environment variables in scripts.

# `(RAG.ipynb)`
Purpose
Detailed demonstration of a retrieval-augmented generation pipeline for PubMedQA:
- Embed contexts with BioGPT
- Index into ChromaDB
- Retrieve contexts for questions
- Generate answers with Azure OpenAI
- Evaluate using BERTScore and LLM-based metrics (RAGAS-style)

Key functions / cells to inspect
- embed_texts_biogpt(texts, batch_size=...)
- retrieve_context(question, k=5)
- generate_answer(question, context)
- BERTScore evaluation and CSV/XLSX export
- RAGAS-style LLM evaluator (async llm_score and prompts)

# `rulebased_eval`
Purpose
Notebooks implementing deterministic rule-based evaluation heuristics for QA outputs. Useful as baseline comparisons to learned models.

Files
- rulebase_base_eval.ipynb — rule-based baseline evaluation on base model outputs
- rulebase_ft_eval.ipynb — rule-based evaluation for fine-tuned outputs

# `frontend`
Purpose
Minimal demo of a frontend and backend for model/QA interaction.

Files
- app.py — frontend application (streamlit)
- backend.py — backend API or helper logic

Run (example)
  ```powershell
  streamlit run app.py
  ```
# `finetune_models`
Purpose
Notebooks demonstrating model fine-tuning strategies for biomedical QA.

Files
- instruction_finetuning.ipynb — instruction-following fine-tuning experiments
- LORA_finetuning.ipynb — parameter-efficient LoRA fine-tuning experiments

# `finetuned_model_error_analysis`
Purpose
Notebooks to analyze errors and failure modes for different finetuned variants.

Files
- base_error_analysis.ipynb — analyze base model errors
- base_instruction_analysis.ipynb — instruction-tuned model analysis
- base_lora_analysis.ipynb — LoRA-tuned model analysis

# `bertscore_eval`
Purpose
Notebooks to compute semantic evaluation using BERTScore across model variants.

Files
- biogpt_base_eval_bertscore.ipynb
- biogpt_base_instruction_bertscore.ipynb
- biogpt_lora_eval_bertscore.ipynb





