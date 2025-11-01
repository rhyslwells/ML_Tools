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
                    "filename": f,
                    "path": os.path.join(root, f)
                })
    return pd.DataFrame(rows)

def fetch_lookup(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ Lookup file not found: {csv_path}")
        return pd.DataFrame(columns=["filename", "pagename"])
    df = pd.read_csv(csv_path)
    if not {"filename", "pagename"}.issubset(df.columns):
        raise ValueError("CSV must contain 'filename' and 'pagename' columns.")
    return df[["filename", "pagename"]]

def build_markdown(df, base_url):
    md = "| File | Path | Page Link |\n|------|-------|------------|\n"
    for _, row in df.iterrows():
        hyperlink = f"[{row['filename']}]({base_url}{row['path']})"
        page_link = f"[[{row['pagename']}]]" if pd.notna(row['pagename']) else ""
        md += f"| {row['filename']} | {hyperlink} | {page_link} |\n"
    return md

if __name__ == "__main__":
    cfg = load_config()
    
    repo_df = list_repo_files(".")
    lookup_df = fetch_lookup(cfg["lookup_csv"])  # CSV defined in config
    merged = repo_df.merge(lookup_df, on="filename", how="left").dropna(subset=["pagename"])
    
    markdown = build_markdown(merged, cfg["base_url"])

    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Markdown table generated and saved to {cfg['output_file']}")
