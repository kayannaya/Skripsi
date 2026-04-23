import argparse
import re
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm


# configurations

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password_here"

DEFAULT_OUTPUT_DIR = r"C:\Users\Design\Desktop\Kayla\Uni\Skripsi\dataConstruction\deconstructed_kg"


# queries

QUERY_STUDIES = """
MATCH (s:Study)
RETURN elementId(s) AS id, labels(s) AS labels, properties(s) AS props
"""

QUERY_STUDY_NEIGHBORHOOD = """
MATCH (s:Study)
WHERE elementId(s) = $study_id
OPTIONAL MATCH (s)-[r]->(n)
RETURN
    type(r)       AS rel_type,
    'outgoing'    AS direction,
    labels(n)     AS target_labels,
    properties(n) AS target_props,
    properties(r) AS rel_props
UNION
MATCH (s:Study)
WHERE elementId(s) = $study_id
OPTIONAL MATCH (n)-[r]->(s)
RETURN
    type(r)       AS rel_type,
    'incoming'    AS direction,
    labels(n)     AS target_labels,
    properties(n) AS target_props,
    properties(r) AS rel_props
"""

QUERY_SCHEMA    = "CALL db.labels() YIELD label RETURN label"
QUERY_REL_TYPES = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"

QUERY_ALL_NODES = """
MATCH (n)
RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
LIMIT 5000
"""

QUERY_ALL_RELS = """
MATCH (a)-[r]->(b)
RETURN
    elementId(a)  AS from_id,
    labels(a)     AS from_labels,
    properties(a) AS from_props,
    type(r)       AS rel_type,
    properties(r) AS rel_props,
    elementId(b)  AS to_id,
    labels(b)     AS to_labels,
    properties(b) AS to_props
LIMIT 50000
"""


# helpers

def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", str(text)).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:max_len] or "unnamed"


def props_to_md(props: dict, exclude: set = None) -> str:
    exclude = exclude or set()
    lines = [
        f"- **{k}**: {v}"
        for k, v in props.items()
        if k not in exclude and v not in (None, "")
    ]
    return "\n".join(lines) if lines else "_No additional properties._"


# markdown renderers

PICO_ORDER = [
    "OUTGOING [HAS_POPULATION]",
    "OUTGOING [HAS_CONDITION]",
    "OUTGOING [HAS_INTERVENTION]",
    "OUTGOING [HAS_COMPARISON]",
    "OUTGOING [HAS_OUTCOME]",
    "OUTGOING [HAS_RESULT]",
    "OUTGOING [BELONGS_TO]",
    "INCOMING [BELONGS_TO]",
]

PICO_HEADINGS = {
    "OUTGOING [HAS_POPULATION]":   "## Population (P)",
    "OUTGOING [HAS_CONDITION]":    "## Condition / Disease",
    "OUTGOING [HAS_INTERVENTION]": "## Intervention (I)",
    "OUTGOING [HAS_COMPARISON]":   "## Comparison (C)",
    "OUTGOING [HAS_OUTCOME]":      "## Outcome (O)",
    "OUTGOING [HAS_RESULT]":       "## Results",
}


def render_study_doc(study_id: str, study_props: dict, neighborhood: list) -> str:
    title = (
        study_props.get("title")
        or study_props.get("name")
        or study_props.get("nctId")
        or study_id
    )

    lines = [
        f"# Clinical Study: {title}",
        "",
        "## Study Metadata",
        props_to_md(study_props, exclude={"title", "name"}),
        "",
    ]

    sections: dict[str, list] = {}
    for row in neighborhood:
        if row["target_labels"] is None:
            continue
        key = f"{row['direction'].upper()} [{row['rel_type']}]"
        sections.setdefault(key, []).append(row)

    ordered_keys = PICO_ORDER + [k for k in sections if k not in PICO_ORDER]

    for key in ordered_keys:
        if key not in sections:
            continue
        heading = PICO_HEADINGS.get(key, f"## {key.replace('_', ' ').title()}")
        lines.append(heading)
        for row in sections[key]:
            node_label = (row["target_labels"] or ["Unknown"])[0]
            props      = row["target_props"] or {}
            rel_props  = row["rel_props"] or {}
            name = (
                props.get("name") or props.get("title")
                or props.get("value") or props.get("text")
                or node_label
            )
            lines.append(f"\n### {node_label}: {name}")
            lines.append(props_to_md(props, exclude={"name", "title", "value", "text"}))
            if rel_props:
                lines.append(f"\n_Relationship properties: {rel_props}_")
        lines.append("")

    lines += ["---", f"_Source: PICO Biomedical Knowledge Graph | Node ID: {study_id}_"]
    return "\n".join(lines)


def render_generic_doc(entity_id: str, labels: list, props: dict, relationships: list) -> str:
    label = (labels or ["Entity"])[0]
    name  = props.get("name") or props.get("title") or props.get("id") or entity_id

    lines = [
        f"# {label}: {name}",
        "",
        "## Properties",
        props_to_md(props),
        "",
        "## Relationships",
    ]

    for r in relationships:
        other_labels = r.get("other_labels") or ["Node"]
        other_props  = r.get("other_props") or {}
        other_name   = (
            other_props.get("name") or other_props.get("title")
            or other_props.get("id") or other_labels[0]
        )
        lines.append(
            f"- **{r.get('direction', '→')} [{r.get('rel_type', 'RELATED_TO')}]**"
            f" → {other_labels[0]}: {other_name}"
        )

    lines += ["", "---", f"_Source: PICO Biomedical Knowledge Graph | Node ID: {entity_id}_"]
    return "\n".join(lines)


# deconstruction pipelines

def run_study_pipeline(driver, output_dir: Path):
    with driver.session() as session:
        studies = list(session.run(QUERY_STUDIES))

    print(f"[INFO] Found {len(studies)} Study nodes")
    total = len(studies)

    for i, record in enumerate(tqdm(studies, desc="Deconstructing studies"), 1):
        study_id    = record["id"]
        study_props = record["props"]

        with driver.session() as session:
            neighborhood = list(session.run(
                QUERY_STUDY_NEIGHBORHOOD, study_id=study_id
            ))

        md_content = render_study_doc(study_id, study_props, neighborhood)

        title    = study_props.get("title") or study_props.get("name") or study_props.get("nctId") or study_id
        filename = safe_filename(title) + ".md"
        (output_dir / filename).write_text(md_content, encoding="utf-8")

        print(f"[{i:3d}/{total}] Written: {filename}")

    print(f"\n[DONE] Deconstructed {total} study documents → {output_dir}/")


def run_generic_pipeline(driver, output_dir: Path):
    with driver.session() as session:
        nodes = {r["id"]: r for r in session.run(QUERY_ALL_NODES)}
        rels  = list(session.run(QUERY_ALL_RELS))

    rel_index: dict[str, list] = defaultdict(list)
    for r in rels:
        rel_index[r["from_id"]].append({
            "direction":    "→",
            "rel_type":     r["rel_type"],
            "other_labels": r["to_labels"],
            "other_props":  r["to_props"],
        })
        rel_index[r["to_id"]].append({
            "direction":    "←",
            "rel_type":     r["rel_type"],
            "other_labels": r["from_labels"],
            "other_props":  r["from_props"],
        })

    print(f"[INFO] Found {len(nodes)} nodes (generic mode)")
    total = len(nodes)

    for i, (nid, record) in enumerate(tqdm(nodes.items(), desc="Deconstructing nodes"), 1):
        md_content = render_generic_doc(
            entity_id     = nid,
            labels        = record["labels"],
            props         = record["props"],
            relationships = rel_index.get(nid, []),
        )
        label    = (record["labels"] or ["node"])[0]
        props    = record["props"]
        name     = props.get("name") or props.get("title") or props.get("id") or nid
        filename = safe_filename(f"{label}_{name}") + ".md"
        (output_dir / filename).write_text(md_content, encoding="utf-8")

        print(f"[{i:3d}/{total}] Written: {filename}")

    print(f"\n[DONE] Deconstructed {total} node documents → {output_dir}/")


def inspect_schema(driver):
    with driver.session() as session:
        labels    = [r["label"] for r in session.run(QUERY_SCHEMA)]
        rel_types = [r["relationshipType"] for r in session.run(QUERY_REL_TYPES)]
    print(f"[INFO] Node labels:        {labels}")
    print(f"[INFO] Relationship types: {rel_types}")
    return labels


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--uri",        type=str, default=NEO4J_URI)
    parser.add_argument("--user",       type=str, default=NEO4J_USER)
    parser.add_argument("--password",   type=str, default=NEO4J_PASSWORD)
    args, _ = parser.parse_known_args()

    # output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # connect to neo4j
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        driver.verify_connectivity()
        print("[INFO] Connected to Neo4j")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Neo4j: {e}")
        print("        Make sure Neo4j is running and the dump has been loaded.")
        return

    # inspect schema and pick pipeline
    labels = inspect_schema(driver)

    if "Study" in labels:
        run_study_pipeline(driver, output_dir)
    else:
        print("[INFO] No 'Study' label found — falling back to generic node pipeline")
        run_generic_pipeline(driver, output_dir)

    driver.close()
    print(f"\n✓ Deconstruction complete. Feed the .md files in {output_dir}/ into LightRAG.")


if __name__ == "__main__":
    main()