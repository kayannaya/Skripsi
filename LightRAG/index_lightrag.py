import argparse
import asyncio
import os
from pathlib import Path

from huggingface_hub import login


# configurations

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "Henrychur/MMed-Llama-3-8B"

DEFAULT_DOCS_DIR  = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\dataConstruction\disease_docs"
DEFAULT_INDEX_DIR = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\LightRAG\lightrag_index"

HF_USERNAME = "kayannaya"


# LightRAG setup

def build_lightrag(index_dir: str, hf_token: str):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=LLM_MODEL, token=hf_token)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content

    async def embed_func(texts: list[str]) -> np.ndarray:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=EMBEDDING_MODEL, token=hf_token)
        embeddings = []
        for text in texts:
            emb = client.feature_extraction(text)
            if isinstance(emb[0], list):
                emb = emb[0]
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    # probe embedding dimension — use new_event_loop() for Python 3.14 compatibility
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sample_emb = loop.run_until_complete(embed_func(["probe"]))
    embed_dim = sample_emb.shape[1]
    print(f"[INFO] Embedding dimension: {embed_dim}")

    rag = LightRAG(
        working_dir=index_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=512,
            func=embed_func,
        ),
    )
    return rag


# document loading

def load_markdown_files(docs_dir: str) -> list[tuple[str, str]]:
    docs_path = Path(docs_dir)
    md_files  = sorted(docs_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in {docs_dir}. "
            "Run generate_disease_docs.py first."
        )
    print(f"[INFO] Found {len(md_files)} Markdown files in {docs_dir}")
    docs = []
    for p in md_files:
        text = p.read_text(encoding="utf-8")
        docs.append((p.stem, text))
    return docs


# indexing

async def insert_documents(rag, docs: list[tuple[str, str]]):
    await rag.initialize_storages()
    total = len(docs)
    for i, (disease_name, text) in enumerate(docs, 1):
        print(f"[{i:3d}/{total}] Indexing: {disease_name}")
        await rag.ainsert(text)
    print(f"\n[DONE] Indexed {total} disease documents")


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir",  type=str, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--hf_token",  type=str, default=None)
    args, _ = parser.parse_known_args()

    # authentication
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or input(
        "Enter your HuggingFace token: "
    )
    login(token=hf_token)

    # index directory
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] LightRAG index will be saved to: {args.index_dir}")

    # build rag instance
    rag = build_lightrag(args.index_dir, hf_token)

    # load and index documents
    docs = load_markdown_files(args.docs_dir)
    asyncio.run(insert_documents(rag, docs))

    print(f"\n[DONE] {len(docs)} documents indexed → {args.index_dir}/")
    print("✓ Indexing complete. Run infer_lightrag.py to query the index.")


if __name__ == "__main__":
    main()