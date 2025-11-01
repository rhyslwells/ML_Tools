import os
import pandas as pd
import yaml

def load_config(path=".mltools_config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def list_repo_files(base_dir="."):
    rows = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".ipynb", ".py")):
                rows.append({
                    "filename": os.path.splitext(f)[0],  # match CSV filenames without extension
                    "path": os.path.join(root, f).replace("\\", "/")  # use forward slashes for GitHub URLs
                })
    return pd.DataFrame(rows)

def fetch_lookup(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ Lookup file not found: {csv_path}")
        return pd.DataFrame(columns=["FileName", "PageName"])
    df = pd.read_csv(csv_path)
    if not {"FileName", "PageName"}.issubset(df.columns):
        raise ValueError("CSV must contain 'FileName' and 'PageName' columns.")
    return df[["FileName", "PageName"]]

def build_markdown(df, base_url):
    md = "| Page | Link |\n|------|------|\n"
    for _, row in df.iterrows():
        page_name = row["PageName"]
        full_url = f"{base_url}{row['path']}"
        md += f"| [{page_name}] | [{page_name}]({full_url}) |\n"
    return md

if __name__ == "__main__":
    cfg = load_config()

    # List repo files and fetch lookup table
    repo_df = list_repo_files(".")
    lookup_df = fetch_lookup(cfg["lookup_csv"])

    # Merge on filename (matching CSV FileName to repo filename without extension)
    merged = repo_df.merge(lookup_df, left_on="filename", right_on="FileName", how="inner")

    # Build markdown table
    markdown = build_markdown(merged, cfg["base_url"])

    # Write output
    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Markdown table generated and saved to {cfg['output_file']}")
