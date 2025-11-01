### **Markdown Table Generator for Repo Files**

**Purpose:**
This script scans a repository for `.py` and `.ipynb` files, merges them with a lookup CSV to map file names to page names, and generates a Markdown table with:

* **Column 1:** Obsidian-style page links (`[[PageName]]`)
* **Column 2:** Clickable file links (`[filename](GitHub URL)`)

**Workflow:**

1. Load configuration from `.mltools_config.yml` (contains `lookup_csv`, `base_url`, and `output_file`).
2. Walk through the repository to list all `.py` and `.ipynb` files.
3. Read the lookup CSV (from the Google Sheet export) containing `FileName` → `PageName` mapping.
4. Merge repo file list with the lookup CSV.
5. Build a Markdown table and save it to the configured output file.

**CSV Source:**
Exported from this Google Sheet:
[Google Sheet link](https://docs.google.com/spreadsheets/d/1bdX6rqy7HZx-hNVdKDHLzU6nAKooZi3rrwIXBfLsRjA/edit?gid=0#gid=0)

**Output:**
A Markdown table suitable for Obsidian or GitHub with file links.
