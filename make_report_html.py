#!/usr/bin/env python3
"""Render the progress reports (EN + ZH markdown) to self-contained, print-friendly HTML.
The figure is embedded as a base64 data URI so each HTML file is fully portable.
Run from repo root: python make_report_html.py
"""
import base64, os, re
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results/extraction_comparison/figs/report_fig.png")

# embed the figure as a data URI
with open(FIG, "rb") as f:
    FIG_DATA = "data:image/png;base64," + base64.b64encode(f.read()).decode()

CSS = """
@page { size: A4; margin: 18mm; }
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Georgia,serif;
 max-width:820px;margin:0 auto;padding:32px 30px 60px;color:#1a1d21;line-height:1.5;font-size:14px;background:#fff}
h1{font-size:22px;line-height:1.25;margin:0 0 14px;color:#0b3d63;border-bottom:3px solid #0b3d63;padding-bottom:10px}
h2{font-size:15.5px;margin:20px 0 7px;color:#0b3d63;border-bottom:1px solid #d8dde3;padding-bottom:3px}
h3{font-size:13.5px;margin:14px 0 5px;color:#23527a}
p{margin:7px 0}
ul,ol{margin:6px 0 6px 4px;padding-left:22px}
li{margin:3px 0}
strong{color:#11212e}
code{background:#f2f4f7;border:1px solid #e3e8ee;border-radius:4px;padding:0 4px;font-size:12.5px;font-family:"SF Mono",Consolas,monospace}
em{color:#33414d}
figure{margin:14px 0;text-align:center}
figure img{max-width:100%;border:1px solid #d8dde3;border-radius:6px}
.abstract{background:#f6f8fb;border:1px solid #d8dde3;border-left:4px solid #0b3d63;border-radius:6px;padding:12px 16px;margin:10px 0 4px}
.abstract h2{border:none;margin:0 0 4px;font-size:14px}
hr{border:none;border-top:1px solid #e3e8ee;margin:18px 0}
a{color:#0b3d63}
.refs{font-size:12.5px;color:#444}
@media print{body{max-width:none;padding:0;font-size:11.5px}h1{font-size:18px}h2{font-size:13px}figure img{max-width:100%}}
"""


def md_to_html(md_path):
    text = open(md_path).read()
    # swap the markdown image reference for the embedded data URI
    text = re.sub(r'!\[([^\]]*)\]\([^)]*report_fig\.png\)',
                  f'![\\1]({FIG_DATA})', text)
    html_body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    # wrap the abstract section (first h2 named Abstract / 摘要) for light styling
    html_body = re.sub(r'(<h2>(?:Abstract|摘要[^<]*)</h2>)', r'<div class="abstract">\1', html_body, count=1)
    # close the abstract div before the next h2
    html_body = html_body.replace('<h2>Accomplishments', '</div><h2>Accomplishments', 1)
    html_body = html_body.replace('<h2>工作成果', '</div><h2>工作成果', 1)
    # tag the references block
    html_body = re.sub(r'(<h2>(?:References|参考文献[^<]*)</h2>)', r'<div class="refs">\1', html_body, count=1)
    if '<div class="refs">' in html_body:
        html_body += '</div>'
    title = "Progress Report" if md_path.endswith("zh.md") is False else "进展报告"
    lang = "zh" if md_path.endswith("zh.md") else "en"
    return f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body>
{html_body}
</body></html>"""


for src, out in [("PROGRESS_REPORT.md", "PROGRESS_REPORT.html"),
                 ("PROGRESS_REPORT_zh.md", "PROGRESS_REPORT_zh.html"),
                 ("MECCC_Progress_Report.md", "MECCC_Progress_Report.html")]:
    html = md_to_html(os.path.join(HERE, src))
    with open(os.path.join(HERE, out), "w") as f:
        f.write(html)
    print("wrote", out, f"({len(html)//1024} KB)")
