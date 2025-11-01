import os
import pandas as pd
import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build

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

def fetch_lookup(sheet_url, sheet_name, creds_path="google_creds.json"):
    creds = service_account.Credentials.from_service_account_file(creds_path)
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    service = build("sheets", "v4", credentials=creds)
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=sheet_name
    ).execute()
    rows = result.get("values", [])
    if not rows:
        return pd.DataFrame(columns=["filename", "pagename"])
    df = pd.DataFrame(rows[1:], columns=rows[0])  # assumes headers exist
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
    sheet_df = fetch_lookup(cfg["sheet_url"], cfg["sheet_name"])
    merged = repo_df.merge(sheet_df, on="filename", how="left").dropna(subset=["pagename"])
    markdown = build_markdown(merged, cfg["base_url"])

    with open(cfg["output_file"], "w") as f:
        f.write(markdown)
