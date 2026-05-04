import argparse
import asyncio
import os
from pathlib import Path


# configurations

# vLLM server settings
LLM_BINDING_HOST        = "https://rtlab-ai-qwen.nomaden.cloud"          
LLM_MODEL               = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LLM_BINDING_API_KEY     = "not-needed-for-vllm"

EMBEDDING_BINDING_HOST  = "https://rtlab-ai-bge.nomaden.cloud"      
EMBEDDING_MODEL         = "BAAI/bge-m3"
EMBEDDING_BINDING_API_KEY = "not-needed-for-vllm"
EMBEDDING_DIM           = 1024

# LightRAG settings
MAX_ASYNC               = 16
MAX_PARALLEL_INSERT     = 5
CHUNK_TOKEN_SIZE        = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100

DEFAULT_DOCS_DIR  = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\dataConstruction\disease_docs"
DEFAULT_INDEX_DIR = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\LightRAG\lightrag_index"


# LightRAG setup

def build_lightrag(index_dir: str):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    import numpy as np
    import httpx

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=999999) as client:
            response = await client.post(
                f"{LLM_BINDING_HOST}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_BINDING_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def embed_func(texts: list[str]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{EMBEDDING_BINDING_HOST}/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {EMBEDDING_BINDING_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()["data"]
            embeddings = [item["embedding"] for item in data]
            return np.array(embeddings, dtype=np.float32)

    rag = LightRAG(
        working_dir=index_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            func=embed_func,
        ),
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        llm_model_max_async=MAX_ASYNC,
        addon_params={
            "language": "Indonesian",
        }
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
    args, _ = parser.parse_known_args()

    # index directory
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] LightRAG index will be saved to: {args.index_dir}")

    # build rag instance
    rag = build_lightrag(args.index_dir)

    # load and index documents
    docs = load_markdown_files(args.docs_dir)
    asyncio.run(insert_documents(rag, docs))

    print(f"\n[DONE] {len(docs)} documents indexed → {args.index_dir}/")
    print("✓ Indexing complete.")


if __name__ == "__main__":
    main()