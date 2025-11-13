# DSA4213_Group19

## Overview
entride centralizes all data-related code for the project:
- Download and store raw datasets
- Clean, normalize, and export processed datasets
- Produce/consume embeddings for retrieval (optional)
- Provide example notebooks and scripts to run RAG / QA experiments

## Requirements
A minimal set of Python packages is listed in `requirement.txt` at the repository root. Example (already present in your repo):
- python>=3.8
- numpy, pandas, scikit-learn, tqdm, pyyaml, python-dotenv
- transformers, torch, sentence-transformers, chromadb, datasets
- openai, bert-score, fastapi, uvicorn
- pytest, flake8, black


## Installation (Windows / PowerShell)
Open PowerShell in the entride folder:

1. Create and activate a virtual environment
```powershell
cd "c:\Users\Jasper\Desktop\DSA4213\entride"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies
```powershell
# if file is named requirement.txt at repo root
pip install -r ..\requirement.txt

# or if you copy/rename it to this folder as requirements.txt
pip install -r requirements.txt
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

## Data formats & locations
- Raw downloads: data/raw/*.csv or *.json
- Processed data: data/processed/*.jsonl (one JSON object per line)
- Expected fields (adjust to scripts):
  - id, question, context (title + abstract), short_answer, long_answer, metadata

Use JSONL for downstream pipelines (easy streaming and safe diffs).

## Typical workflows
1. Download raw data (see above).
2. Run preprocessing script to normalize text, combine fields, dedupe and export JSONL:
```powershell
python src/preprocess_pubmedqa.py --input data/raw/PubMedQA_pqa_labeled_train.csv --output data/processed/train.jsonl
```
3. (Optional) Generate embeddings for retrieval:
```powershell
python src/generate_embeddings.py --input data/processed/train.jsonl --output data/embeddings/train_embeddings.npy
```
4. Open notebooks/RAG.ipynb to index contexts and run retrieval + generation experiments.

Adjust script names and CLI flags to match the files under src/.

## Configuration & environment variables
Keep secrets out of VCS (use .env). Typical variables:
- AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT
- HF_TOKEN (if accessing private Hugging Face resources)
Use python-dotenv or OS environment variables in scripts.

## Development & testing
- Run tests:
```powershell
pytest -q
```
- Lint and format:
```powershell
black src tests
flake8 src tests
```

## Notes & best practices
- Keep data/raw immutable. Re-run preprocessing to produce processed outputs.
- Log preprocessing steps and parameters (save small metadata json next to processed files).
- Pin dependency versions in requirement file and update deliberately (pip freeze > requirement.txt).

## License & contact
Add a LICENSE file at the repository root. For questions, contact the project owner or course staff.

References
- PubMedQA dataset (Hugging Face): https://huggingface.co/datasets/qiaojin/PubMedQA
- Hugging Face datasets docs: https://huggingface.co/docs/datasets
