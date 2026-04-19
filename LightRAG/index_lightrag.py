import argparse
import asyncio
import os
import lightrag
from pathlib import Path

from huggingface_hub import login


# configurations

# hf interference model 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL       = "Henrychur/MMed-Llama-3-8B"            

DEFAULT_DOCS_DIR  = "./disease_docs"
DEFAULT_INDEX_DIR = "./lightrag_index"

HF_USERNAME = "kayannaya"


# LightRAG setup

def build_lightrag(index_dir: str, hf_token: str):
    """
    Initialise a LightRAG instance backed by HuggingFace Inference.

    LightRAG needs two callables:
      - llm_model_func   : text generation  (for graph entity extraction)
      - embedding_func   : text→vector      (for vector retrieval)
    """
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.hf import hf_model_complete, hf_embed
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        """Thin async wrapper around HF Inference text generation."""
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
        """Thin async wrapper around HF Inference feature extraction."""
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=EMBEDDING_MODEL, token=hf_token)
        embeddings = []
        for text in texts:
            emb = client.feature_extraction(text)
            
            if isinstance(emb[0], list):
                emb = emb[0]           
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    # probe embedding dimension
    sample_emb = asyncio.get_event_loop().run_until_complete(
        embed_func(["probe"])
    )
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
    """
    Returns a list of (disease_name, markdown_text) tuples.
    """
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
    """
    Insert each disease document into LightRAG.
    LightRAG handles chunking, entity extraction, and graph construction.
    """
    total = len(docs)
    for i, (disease_name, text) in enumerate(docs, 1):
        print(f"[{i:3d}/{total}] Indexing: {disease_name}")
        await rag.ainsert(text)
    print(f"\n[DONE] Indexed {total} disease documents → LightRAG working dir")


# main

def main():
    parser = argparse.ArgumentParser(
        description="Index disease Markdown files into LightRAG."
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=DEFAULT_DOCS_DIR,
        help="Directory containing per-disease .md files",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="LightRAG working directory (graph + vector store will be saved here)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    args, _ = parser.parse_known_args()

    # authentication of hf
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or input(
        "Enter your HuggingFace token: "
    )
    login(token=hf_token)

    # index direction
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] LightRAG index will be saved to: {args.index_dir}")

    # build rag instance
    rag = build_lightrag(args.index_dir, hf_token)

    # loading documents
    docs = load_markdown_files(args.docs_dir)
    asyncio.get_event_loop().run_until_complete(insert_documents(rag, docs))

    print("\n── Index summary ────────────────────────────────────────────────")
    print(f"   Documents indexed : {len(docs)}")
    print(f"   Index location    : {args.index_dir}/")
    print("   Files created     : graph_chunk_entity_relation.graphml,")
    print("                       kv_store_*.json, vdb_*.json")
    print("\n✓ Indexing complete. Run infer_lightrag.py to query the index.")


if __name__ == "__main__":
    main()