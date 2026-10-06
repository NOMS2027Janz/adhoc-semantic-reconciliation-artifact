#!/usr/bin/env python3
"""Render selected Markdown reports to print-ready PDFs (md -> HTML -> wkhtmltopdf)."""
import subprocess, sys, tempfile, os
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
# The README (repo root) plus the five reports under reports/ — the master synthesis and the four
# setting reports. The setting-1 sub-studies are repo-only method notes under notes/studies/
# (their essentials are folded into reports/REPORT_1of4_configuration.md) and are not PDF'd.
DOCS = ["README.md",
        "reports/MASTER_REPORT.md",
        "reports/REPORT_1of4_configuration.md",
        "reports/REPORT_2of4_intent.md",
        "reports/REPORT_3of4_standard_free.md",
        "reports/REPORT_4of4_observability.md"]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 12pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 23pt; line-height: 1.2; margin: 0 0 .4em; border-bottom: 2px solid #2a78d6;
     padding-bottom: .2em; }
h2 { font-size: 16.5pt; margin: 1.4em 0 .4em; color: #14314f; }
h3 { font-size: 13.5pt; margin: 1.1em 0 .3em; color: #14314f; }
p, li { font-size: 12pt; }
code, pre { font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: 10.5pt; }
pre { background: #f5f4f2; border: 1px solid #e3e1dd; border-radius: 4px;
      padding: .7em .9em; white-space: pre-wrap; word-wrap: break-word; }
code { background: #f0eeea; padding: .05em .3em; border-radius: 3px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #eb6834; margin: .8em 0; padding: .3em 1em;
             background: #fbf6f2; color: #333; font-style: italic; }
blockquote p { margin: .3em 0; }
table { border-collapse: collapse; margin: 1em 0; font-size: 11pt; width: auto; }
th, td { border: 1px solid #cfcdc8; padding: .4em .65em; text-align: left; }
th { background: #eef3fa; }
td[align="center"], th[align="center"] { text-align: center; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
a { color: #2a78d6; text-decoration: none; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
h1, h2, h3 { page-break-after: avoid; }
table, pre, blockquote, img { page-break-inside: avoid; }
"""

HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>{css}</style></head><body>{body}</body></html>"""


def build(md_name):
    src = ROOT / md_name
    text = src.read_text()
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    html = HTML.format(css=CSS, body=body)
    # Write the intermediate HTML and the PDF next to the source .md, and run wkhtmltopdf from that
    # directory, so relative image paths (e.g. ../figures/…) resolve from the report's own location.
    html_path = src.with_suffix(".print.html")
    html_path.write_text(html)
    pdf_path = src.with_suffix(".pdf")
    subprocess.run(
        ["wkhtmltopdf", "--enable-local-file-access", "--quiet",
         "--print-media-type", "--dpi", "150",
         "--margin-top", "18mm", "--margin-bottom", "18mm",
         "--margin-left", "16mm", "--margin-right", "16mm",
         str(html_path), str(pdf_path)],
        check=True, cwd=str(src.parent),
    )
    html_path.unlink()
    print("wrote", pdf_path.name, f"({pdf_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    for d in DOCS:
        build(d)
