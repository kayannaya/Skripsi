import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from huggingface_hub import login


# configurations

MODEL_PATH  = "Henrychur/MMed-Llama-3-8B"
INDEX_DIR   = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\LightRAG\lightrag_index"
OUTPUT_DIR  = "./lightrag-inference-output"

HF_USERNAME     = "kayannaya"
TRACKIO_SPACE   = f"{HF_USERNAME}/qlora-tracking"
TRACKIO_PROJECT = "train-qlora"
RUN_NAME        = "run-01_lightrag_hybrid"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS  = 300
SEED            = 42

# retrieval modes: naive, local, global, hybrid
RETRIEVAL_MODE = "hybrid"

# pico guardrail injected into every prompt
PICO_GUARDRAIL = (
    "Anda adalah asisten medis klinis. "
    "Jawab pertanyaan berdasarkan bukti klinis yang relevan. "
    "Strukturkan jawaban Anda menggunakan kerangka PICO bila memungkinkan: "
    "Population (Populasi), Intervention (Intervensi), "
    "Comparison (Perbandingan), Outcome (Luaran)."
)


# LightRAG setup

def build_lightrag(index_dir: str, hf_token: str):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # load embedding model locally — no api calls
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=MODEL_PATH, token=hf_token)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history_messages:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})
        response = client.chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def embed_func(texts: list[str]) -> np.ndarray:
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    # probe embedding dimension — new_event_loop() for Python 3.14 compatibility
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    sample_emb = loop.run_until_complete(embed_func(["probe"]))
    embed_dim  = sample_emb.shape[1]

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


# data loading

def load_scenarios(filepath: str) -> list[dict]:
    path = Path(filepath)
    scenarios = []

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    scenarios.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed line: {e}")
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenarios.append(dict(row))

    print(f"[INFO] Loaded {len(scenarios)} scenarios from {filepath}")
    return scenarios


# prompt formatting

def format_prompt(example: dict, with_guardrail: bool = True) -> str:
    instruction  = example.get("instruction", "").strip()
    user_input   = example.get("input", "").strip()
    system_block = PICO_GUARDRAIL if with_guardrail else ""
    user_message = f"{instruction}\n\n{user_input}"

    prompt = (
        f"<|im_start|>system\n{system_block}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# generate response

async def generate_response_async(rag, instruction: str, pico_input: str, mode: str = RETRIEVAL_MODE) -> str:
    from lightrag import QueryParam

    await rag.initialize_storages()
    query    = f"{instruction}\n\n{pico_input}"
    response = await rag.aquery(
        query,
        param=QueryParam(
            mode=mode,
            only_need_context=False,
            response_type="single line"
        ),
    )
    return response


def generate_response(rag, instruction: str, pico_input: str, mode: str = RETRIEVAL_MODE) -> str:
    return asyncio.run(
        generate_response_async(rag, instruction, pico_input, mode)
    )


# batch inference

def run_inference(rag, scenarios: list[dict], output_dir: str, mode: str, run_name: str) -> list[dict]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    print(f"[INFO] Running inference — {len(scenarios)} scenarios | mode: {mode}")

    for i, scenario in enumerate(scenarios, 1):
        instruction  = scenario.get("instruction", "")
        pico_input   = scenario.get("input", "")
        ground_truth = scenario.get("output", "")

        print(f"[{i:3d}/{len(scenarios)}] Generating...")
        try:
            response = generate_response(rag, instruction, pico_input, mode)
        except Exception as e:
            print(f"[ERROR] {e}")
            response = ""

        results.append({
            "id"            : i,
            "instruction"   : instruction,
            "input"         : pico_input,
            "ground_truth"  : ground_truth,
            "generated"     : response,
            "method"        : "lightrag",
            "retrieval_mode": mode,
            "model"         : MODEL_PATH,
            "run_name"      : run_name,
        })

        if i == 1:
            print(f"\ninstruction  : {instruction[:120]}")
            print(f"generated    : {response[:300]}")
            print(f"ground truth : {ground_truth[:300]}\n")

    out_file = Path(output_dir) / f"{run_name}_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Results saved → {out_file}")
    return results


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios",  type=str, default="./pico_scenarios.csv")
    parser.add_argument("--index_dir",  type=str, default=INDEX_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--hf_token",   type=str, default=None)
    parser.add_argument("--mode",       type=str, default=RETRIEVAL_MODE, choices=["naive", "local", "global", "hybrid"])
    parser.add_argument("--run_name",   type=str, default=RUN_NAME)
    args, _ = parser.parse_known_args()

    # authentication
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or input(
        "Enter your HuggingFace token: "
    )
    login(token=hf_token)

    # validate index
    if not Path(args.index_dir).exists():
        raise FileNotFoundError(
            f"Index directory '{args.index_dir}' not found. "
            "Run index_lightrag.py first."
        )

    # build rag
    print(f"[INFO] Loading LightRAG index from {args.index_dir} ...")
    rag = build_lightrag(args.index_dir, hf_token)

    # load scenarios
    scenarios = load_scenarios(args.scenarios)

    # run inference
    results = run_inference(
        rag,
        scenarios,
        output_dir=args.output_dir,
        mode=args.mode,
        run_name=args.run_name,
    )

    total     = len(results)
    non_empty = sum(1 for r in results if r["generated"].strip())
    print(f"\n[DONE] {non_empty}/{total} responses generated → {args.output_dir}/")


if __name__ == "__main__":
    main()