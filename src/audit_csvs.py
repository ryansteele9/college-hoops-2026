"""Audit all CSVs in data/raw/ and write summary to data/processed/csv_audit.csv.
Also updates CLAUDE.md with a ## Data Files section.
"""

import os
import re
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLAUDE_MD = os.path.join(os.path.dirname(__file__), "..", "CLAUDE.md")
AUDIT_OUT = os.path.join(PROCESSED_DIR, "csv_audit.csv")

SPINE_COLS = {"year", "team_no", "team_id", "team", "seed", "round"}


def normalize_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def is_spine(col: str) -> bool:
    return normalize_col(col) in SPINE_COLS


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_paths:
        print("No CSVs found in data/raw/")
        return

    # Step 1: Load all CSVs
    frames = {}
    for path in csv_paths:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path, low_memory=False)
            frames[name] = df
        except Exception as e:
            print(f"WARNING: Could not read {name}: {e}")

    # Step 2 & 3: Build global column map (non-spine columns per file)
    file_extra_cols = {}
    col_file_count = {}
    for name, df in frames.items():
        extra = [c for c in df.columns if not is_spine(c)]
        file_extra_cols[name] = extra
        for c in extra:
            col_file_count[c] = col_file_count.get(c, 0) + 1

    # Step 4: Per-file metrics
    rows = []
    for name, df in frames.items():
        extra = file_extra_cols[name]

        # Years
        year_col = next((c for c in df.columns if normalize_col(c) == "year"), None)
        if year_col is not None:
            year_min = df[year_col].min()
            year_max = df[year_col].max()
        else:
            year_min = year_max = ""

        # Nulls
        total_nulls = int(df.isnull().sum().sum())
        null_detail_parts = []
        for c in df.columns:
            n = int(df[c].isnull().sum())
            if n > 0:
                null_detail_parts.append(f"{c}:{n}")
        null_detail = "; ".join(null_detail_parts)

        # Unique columns (appear in only this file)
        unique_cols = [c for c in extra if col_file_count[c] == 1]

        rows.append({
            "filename": name,
            "row_count": len(df),
            "year_min": year_min,
            "year_max": year_max,
            "extra_columns": ", ".join(extra),
            "total_nulls": total_nulls,
            "null_detail": null_detail,
            "unique_columns": ", ".join(unique_cols),
        })

    # Step 5: Write CSV
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(AUDIT_OUT, index=False)
    print(f"Wrote {AUDIT_OUT} ({len(audit_df)} rows)\n")

    # Step 6: Print readable summary
    for row in rows:
        name = row["filename"]
        rc = row["row_count"]
        ymin, ymax = row["year_min"], row["year_max"]
        extra_list = [c for c in row["extra_columns"].split(", ")] if row["extra_columns"] else []
        nulls = row["total_nulls"]
        unique_list = [c for c in row["unique_columns"].split(", ")] if row["unique_columns"] else []

        year_str = f"{ymin}–{ymax}" if ymin != "" else "N/A"
        print(f"=== {name} ===")
        print(f"  Rows: {rc} | Years: {year_str}")
        extra_preview = ", ".join(extra_list[:6])
        if len(extra_list) > 6:
            extra_preview += f", ... (+{len(extra_list)-6} more)"
        print(f"  Extra cols ({len(extra_list)}): {extra_preview}")
        print(f"  Nulls: {nulls}")
        if unique_list:
            unique_preview = ", ".join(unique_list[:6])
            if len(unique_list) > 6:
                unique_preview += f", ... (+{len(unique_list)-6} more)"
            print(f"  Unique cols: {unique_preview}")
        print()

    # Step 7: Update CLAUDE.md
    md_table_lines = [
        "## Data Files\n",
        "\n",
        "| File | Rows | Years | Extra Columns | Unique Columns |\n",
        "|------|------|-------|---------------|----------------|\n",
    ]
    for row in rows:
        fname = row["filename"]
        rc = row["row_count"]
        ymin, ymax = row["year_min"], row["year_max"]
        year_str = f"{ymin}–{ymax}" if ymin != "" else "N/A"
        extra_list = [c for c in row["extra_columns"].split(", ")] if row["extra_columns"] else []
        unique_list = [c for c in row["unique_columns"].split(", ")] if row["unique_columns"] else []
        extra_preview = ", ".join(extra_list[:4])
        if len(extra_list) > 4:
            extra_preview += f" (+{len(extra_list)-4} more)"
        unique_preview = ", ".join(unique_list[:4])
        if len(unique_list) > 4:
            unique_preview += f" (+{len(unique_list)-4} more)"
        md_table_lines.append(
            f"| {fname} | {rc} | {year_str} | {extra_preview} | {unique_preview} |\n"
        )

    md_table = "".join(md_table_lines)

    with open(CLAUDE_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace existing ## Data Files section or append
    pattern = r"## Data Files\n.*?(?=\n## |\Z)"
    if re.search(pattern, content, flags=re.DOTALL):
        content = re.sub(pattern, md_table.rstrip(), content, flags=re.DOTALL)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + md_table

    with open(CLAUDE_MD, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Updated {CLAUDE_MD} with ## Data Files section ({len(rows)} files).")


if __name__ == "__main__":
    main()
