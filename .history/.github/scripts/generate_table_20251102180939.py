import os
import pandas as pd
import yaml

def load_config(path=".github/.mltools_config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def list_repo_files(base_dir="."):
    rows = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".ipynb", ".py")):
                full_path = os.path.join(root, f).replace("\\", "/")
                # remove leading "./" for cleaner GitHub URLs
                if full_path.startswith("./"):
                    full_path = full_path[2:]
                rows.append({
                    "filename": os.path.splitext(f)[0],  # without extension
                    "path": full_path
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
        filename = row["filename"]  # <-- get file name without extension
        full_url = f"{base_url}{row['path']}"
        md += f"| [[{page_name}]] | [{filename}]({full_url}) |\n"
    return md

if __name__ == "__main__":
    cfg = load_config()

    repo_df = list_repo_files(".")
    # save to CSV for debugging
    # repo_df.to_csv("repo_files.csv", index=False)
    lookup_df = fetch_lookup(cfg["lookup_csv"])

    # sort by page name
    merged = repo_df.merge(lookup_df, left_on="filename", right_on="FileName", how="inner")

    # sort by page name
    merged = merged.sort_values("PageName")

    markdown = build_markdown(merged, cfg["base_url"])

    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"✅ Markdown table generated and saved to {cfg['output_file']}")
