import os
import pandas as pd
import yaml

def load_config(path=".mltools_config.yml"):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_repo_files(base_dir="."):
    """List all .ipynb and .py files recursively in the repo."""
    rows = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".ipynb", ".py")):
                rows.append({
                    "filename": os.path.splitext(f)[0],  # remove extension for matching
                    "path": os.path.join(root, f).replace("\\", "/")  # consistent URL path
                })
    df = pd.DataFrame(rows)
    print(f"📂 Found {len(df)} code files in repository.")
    return df

def fetch_lookup(csv_path):
    """Load lookup CSV with FileName and PageName columns."""
    if not os.path.exists(csv_path):
        print(f"⚠️ Lookup file not found: {csv_path}")
        return pd.DataFrame(columns=["FileName", "PageName"])
    df = pd.read_csv(csv_path)
    if not {"FileName", "PageName"}.issubset(df.columns):
        raise ValueError("CSV must contain 'FileName' and 'PageName' columns.")
    print(f"📖 Loaded {len(df)} lookup entries from {csv_path}.")
    return df[["FileName", "PageName"]]

def build_markdown(df, base_url):
    """Build the Markdown table linking page names to GitHub file paths."""
    md_lines = ["| Page | Link |", "|------|------|"]
    for _, row in df.iterrows():
        page_name = row["PageName"]
        full_url = f"{base_url}{row['path']}"
        md_lines.append(f"| [{page_name}] | [{page_name}]({full_url}) |")
    return "\n".join(md_lines) + "\n"

if __name__ == "__main__":
    cfg = load_config()

    repo_df = list_repo_files(".")
    lookup_df = fetch_lookup(cfg["lookup_csv"])

    # Inner join: only include repo files listed in lookup
    merged = repo_df.merge(lookup_df, left_on="filename", right_on="FileName", how="inner")
    print(f"✅ Matched {len(merged)} files between repo and lookup.")

    markdown = build_markdown(merged, cfg["base_url"])

    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Markdown table generated and saved to {cfg['output_file']}")
