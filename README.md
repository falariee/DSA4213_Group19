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


