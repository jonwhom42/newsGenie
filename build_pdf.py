"""Convert REPORT.md -> REPORT.html (styled) for PDF export via headless Edge.

Run:  python build_pdf.py
Then: msedge --headless=new --print-to-pdf=REPORT.pdf --no-pdf-header-footer file:///.../REPORT.html
"""
import pathlib
import markdown

HERE = pathlib.Path(__file__).parent
md_text = (HERE / "REPORT.md").read_text(encoding="utf-8")

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "codehilite", "sane_lists", "toc"],
    extension_configs={"codehilite": {"noclasses": True, "guess_lang": False}},
)

CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 22pt; border-bottom: 3px solid #2563eb; padding-bottom: 6px; color: #0f172a; }
h2 { font-size: 15pt; margin-top: 22px; border-bottom: 1px solid #cbd5e1;
     padding-bottom: 4px; color: #1e3a8a; page-break-after: avoid; }
h3 { font-size: 12pt; color: #334155; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 11pt; color: #475569; page-break-after: avoid; }
p, li { font-size: 10.5pt; }
a { color: #2563eb; text-decoration: none; }
strong { color: #0f172a; }

code { font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
       font-size: 9pt; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
pre { font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
      font-size: 8.5pt; line-height: 1.35; background: #f8fafc;
      border: 1px solid #e2e8f0; border-left: 3px solid #2563eb;
      border-radius: 4px; padding: 10px 12px; white-space: pre-wrap;
      word-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }

table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
        page-break-inside: auto; }
th, td { border: 1px solid #cbd5e1; padding: 6px 9px; text-align: left;
         vertical-align: top; }
th { background: #eff6ff; color: #1e3a8a; font-weight: 600; }
tr { page-break-inside: avoid; }
tr:nth-child(even) td { background: #f8fafc; }

blockquote { border-left: 4px solid #22c55e; background: #f0fdf4; margin: 12px 0;
             padding: 8px 14px; border-radius: 0 4px 4px 0; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>NewsGenie — Project Report</title>
<style>{CSS}</style></head>
<body>{html_body}</body></html>"""

out = HERE / "REPORT.html"
out.write_text(html, encoding="utf-8")
print("Wrote", out)
