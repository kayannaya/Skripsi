import argparse
import re
import sys
from pathlib import Path
 
import pandas as pd
 
 
# configuration
 
DEFAULT_DATA_PATH = "./database_dokterGPT.csv"
DEFAULT_OUT_DIR   = "./disease_docs"
 
# disease name
DISEASE_COL_IDX = 0
 
# header rename
COLUMN_HEADER_MAP = {
    "Pendahuluan"                  : "Pendahuluan (Latar Belakang)",
    "Definisi"                     : "Definisi",
    "Patogenesis/ Patofisiologi"   : "Patogenesis / Patofisiologi",
    "Faktor Risiko"                : "Faktor Risiko",
    "Anamnesis"                    : "Anamnesis",
    "Pemeriksaan Fisik"            : "Pemeriksaan Fisik",
    "Pemeriksaan Penunjang"        : "Pemeriksaan Penunjang",
    "Tatalaksana Non Farmakologi"  : "Tatalaksana Non-Farmakologi",
    "Tatalaksana Farmakologi"      : "Tatalaksana Farmakologi",
    "Lain-lain"                    : "Informasi Tambahan",
    "Referensi"                    : "Referensi",
}
 
# column names to exclude entirely from output (add as needed).
EXCLUDED_COLS: set[str] = set()
 
 
# helper functions
def slugify(name: str) -> str:
    """Convert a disease name into a safe, lowercase filename stem."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120]   # cap for filesystem safety
 
 
def is_empty(value) -> bool:
    """
    Return True if a cell should be treated as missing.
    Handles: NaN floats, None, empty string, whitespace-only string.
    """
    if value is None:
        return True
    # pandas stores missing cells as float NaN
    try:
        import math
        if math.isnan(float(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""
 
 
def clean_text(value: str) -> str:
    """
    Normalise raw CSV cell text for readable Markdown:
    - Replace literal \\n escape sequences with real newlines.
    - Strip surrounding whitespace.
    - Collapse 3+ consecutive blank lines to 2.
    """
    text = str(value)
    text = text.replace("\\n", "\n")
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
 
 
def detect_content_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all columns except the disease-name column and any excluded ones,
    preserving original CSV order.
    """
    name_col = df.columns[DISEASE_COL_IDX]
    return [
        c for c in df.columns
        if c != name_col and c not in EXCLUDED_COLS
    ]
 
 
# building markdown file
 
def build_markdown(disease_name: str, row: pd.Series, content_cols: list[str]) -> str:
    """
    Build a structured Markdown document for one disease row.
    Sections whose value is null / empty are silently omitted —
    no empty headers, no placeholder text.
    """
    lines: list[str] = []
 
    # Document title
    lines.append(f"# {disease_name.strip()}\n")
 
    sections_written = 0
 
    for col in content_cols:
        value = row.get(col)
 
        # empty guard
        if is_empty(value):
            continue            # silently skip — no empty section left behind
 
        text = clean_text(value)
        if not text:
            continue            # blank after normalisation — also skip
 
        # section header
        header = COLUMN_HEADER_MAP.get(col, col)    # fallback to raw col name
        lines.append(f"## {header}\n")
 
        # section body
        lines.append(f"{text}\n")
        sections_written += 1
 
    # metadata footer
    lines.append("---\n")
    lines.append(f"*Sumber: DokterGPT Database*  ")
    lines.append(f"*Seksi tersedia: {sections_written} dari {len(content_cols)}*\n")
 
    return "\n".join(lines)
 
 
# main
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-disease Markdown files from database_dokterGPT.csv"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the DokterGPT CSV file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for .md files",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the first generated document to stdout and exit without saving",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV encoding (default: utf-8). Try utf-8-sig or latin-1 if text is garbled.",
    )
    # Colab-safe: ignore kernel-injected args like '-f'
    args, _ = parser.parse_known_args()
 
    # load database
    try:
        df = pd.read_csv(args.data, encoding=args.encoding)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(
            f"[ERROR] Encoding error. Try --encoding utf-8-sig or --encoding latin-1",
            file=sys.stderr,
        )
        sys.exit(1)
 
    disease_col   = df.columns[DISEASE_COL_IDX]
    content_cols  = detect_content_columns(df)
    total         = len(df)
 
    print(f"[INFO] Loaded         : {total} rows from {args.data}")
    print(f"[INFO] Disease column : '{disease_col}'")
    print(f"[INFO] Content columns ({len(content_cols)}):")
    for c in content_cols:
        print(f"         • {c}")
    print()
 
    # null summary
    print("── Missing values per column ────────────────────────────────────")
    for col in content_cols:
        n_null = int(df[col].apply(is_empty).sum())
        pct    = 100 * n_null / total
        bar    = "█" * int(pct / 5)
        print(f"  {col:<42} {n_null:3d}/{total} ({pct:5.1f}%)  {bar}")
    print()
 
    # preview mode
    if args.preview:
        first_row    = df.iloc[0]
        disease_name = str(first_row[disease_col])
        md           = build_markdown(disease_name, first_row, content_cols)
        print(f"── Preview: {disease_name} ──────────────────────────────────────")
        print(md[:4000])
        return
 
    # write files
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    written         = 0
    skipped_empty   = 0
    skipped_error   = 0
    slug_seen: dict[str, int] = {}
 
    for idx, row in df.iterrows():
        disease_name = str(row[disease_col]).strip()

        # guards
        # missing disease name
        if not disease_name or disease_name.lower() in ("nan", "none", ""):
            print(f"[WARN] Row {idx}: empty disease name — skipped")
            skipped_empty += 1
            continue
 
        # all content columns empty for this row
        has_content = any(not is_empty(row.get(c)) for c in content_cols)
        if not has_content:
            print(f"[WARN] Row {idx} '{disease_name}': all content columns empty — skipped")
            skipped_empty += 1
            continue
 
        # resolve filename slug
        base_slug = slugify(disease_name)
        if base_slug in slug_seen:
            slug_seen[base_slug] += 1
            slug = f"{base_slug}_{slug_seen[base_slug]}"
        else:
            slug_seen[base_slug] = 0
            slug = base_slug
 
        filepath = out_path / (slug + ".md")
 
        try:
            md = build_markdown(disease_name, row, content_cols)
            filepath.write_text(md, encoding="utf-8")
            written += 1
        except OSError as e:
            print(f"[ERROR] Could not write '{filepath.name}': {e}")
            skipped_error += 1
 
    # final REPORT
    print("── Results ──────────────────────────────────────────────────────")
    print(f"  Written  : {written} Markdown files  →  {out_path}/")
    if skipped_empty:
        print(f"  Skipped  : {skipped_empty} rows (empty name or all-null content)")
    if skipped_error:
        print(f"  Errors   : {skipped_error} files could not be written")
 
    file_list = sorted(out_path.glob("*.md"))
    print(f"\n── File list ({len(file_list)}) ─────────────────────────────────────")
    for p in file_list[:30]:
        kb         = p.stat().st_size / 1024
        n_sections = p.read_text(encoding="utf-8").count("\n## ")
        print(f"  {p.name:<65}  {kb:6.1f} KB  {n_sections} sections")
    if len(file_list) > 30:
        print(f"  ... and {len(file_list) - 30} more")
 
    print(f"\n✓ Done. Next: python index_lightrag.py --docs_dir {out_path}/")
 
 
if __name__ == "__main__":
    main()