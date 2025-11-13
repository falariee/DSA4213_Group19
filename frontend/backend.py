import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModel
import torch

# ============================================
# 0. Load .env (Azure settings)
# ============================================
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ============================================
# 1. Config
# ============================================
input_csv = "PubMedQA_artificial_RAG.csv"
output_file = "pubmedqa_context_embeddings_biogpt.parquet"
persist_dir = "./chroma_biogpt"
collection_name = "pubmedqa_biogpt"
batch_size = 64

# ============================================
# 2. Load BioGPT model (local GPU embeddings)
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for BioGPT embeddings")

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModel.from_pretrained("microsoft/BioGPT").to(device)
model.eval()

@torch.no_grad()
def embed_texts(texts):
    """Compute average pooled embeddings from BioGPT last hidden state."""
    embs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embs.append(last_hidden[0])
    return embs

# ============================================
# 3. Load and expand dataset
# ============================================
df = pd.read_csv(input_csv)

def parse_context(raw_context):
    import ast
    try:
        parsed = ast.literal_eval(raw_context)
        contexts = parsed.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        return [c.strip() for c in contexts]
    except Exception:
        return [raw_context.strip()]

rows = []
for idx, row in df.iterrows():
    contexts = parse_context(row["context"])
    for i, ctx in enumerate(contexts):
        rows.append({
            "id": f"{idx}_ctx{i}",
            "question": row["question"],
            "context": ctx,
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"]
        })
expanded_df = pd.DataFrame(rows)

# ============================================
# 4. Embed with BioGPT and save to Parquet
# ============================================
embeddings = []
for i in tqdm(range(0, len(expanded_df), batch_size), desc="Embedding with BioGPT (GPU)"):
    batch_texts = expanded_df["context"].iloc[i:i+batch_size].tolist()
    batch_embs = embed_texts(batch_texts)
    embeddings.extend(batch_embs)

expanded_df["embedding"] = embeddings
expanded_df.to_parquet(output_file, index=False)
print(f"Saved BioGPT embeddings to {output_file}")

# ============================================
# 5. Store in Chroma
# ============================================
client_chroma = chromadb.PersistentClient(path=persist_dir)
collection = client_chroma.get_or_create_collection(name=collection_name)

ids = expanded_df["id"].astype(str).tolist()
texts = expanded_df["context"].tolist()
metadatas = expanded_df[["question", "long_answer", "final_decision"]].to_dict(orient="records")
embs = embeddings

for i in tqdm(range(0, len(expanded_df), batch_size), desc="Upserting into Chroma"):
    collection.upsert(
        ids=ids[i:i+batch_size],
        documents=texts[i:i+batch_size],
        embeddings=embs[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )
print("All BioGPT embeddings inserted into Chroma")

# ============================================
# 6. Retrieval + Azure GPT Generation
# ============================================
def embed_query_biogpt(query: str):
    return embed_texts([query])[0]

def retrieve_context(query, k=30):
    q_emb = embed_query_biogpt(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    return "\n\n".join(results["documents"][0])

client_aoai = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def generate_answer(query):
    context = retrieve_context(query, k=30)
    prompt = f"""
    You are a medical assistant.
    Use the context below to answer truthfully.
    Give as detailed answers as possible.
    If the answer is not in the context, say "I don’t know."

    Context:
    {context}

    Question: {query}
    """
    response = client_aoai.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ============================================
# 7. Entry point for Streamlit
# ============================================
def get_response(query: str) -> str:
    try:
        return generate_answer(query)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print(get_response("Does artificial intelligence improve medical diagnosis?"))
