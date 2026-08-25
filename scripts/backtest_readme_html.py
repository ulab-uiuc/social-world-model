#!/usr/bin/env python3
"""Render docs/backtest.md to a self-contained, themed HTML page.

Same palette and type as the generated report (scripts/backtest_html.py) so the
prose document and the results dashboard read as one thing. No external assets
beyond Google Fonts, which is the one host an Artifact CSP admits.
"""

import argparse
import re
from pathlib import Path

import markdown

CSS = """
:root {
  --ground:#F1F5F6; --surface:#FFFFFF; --raised:#F7FAFA;
  --ink:#12212A; --body:#2C3D45; --muted:#5E727B; --faint:#8FA1A8;
  --rule:#D6E0E3; --rule-soft:#E6EDEF;
  --accent:#1A5C64; --accent-soft:#E2EFF0; --gold:#9A6E1C;
  --gain:#1F7350; --loss:#A83535;
  --shadow:0 1px 2px rgba(18,33,42,.05), 0 8px 24px -12px rgba(18,33,42,.16);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0C1518; --surface:#141F23; --raised:#1A272C;
    --ink:#E7F0F2; --body:#C3D2D6; --muted:#8DA2A9; --faint:#63787F;
    --rule:#25353A; --rule-soft:#1D2B30;
    --accent:#5FB3BC; --accent-soft:#123336; --gold:#D2A44E;
    --gain:#4FBC8C; --loss:#E37070;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --ground:#0C1518; --surface:#141F23; --raised:#1A272C;
  --ink:#E7F0F2; --body:#C3D2D6; --muted:#8DA2A9; --faint:#63787F;
  --rule:#25353A; --rule-soft:#1D2B30;
  --accent:#5FB3BC; --accent-soft:#123336; --gold:#D2A44E;
  --gain:#4FBC8C; --loss:#E37070;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.7);
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--body);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  font-size:16.5px; line-height:1.66; -webkit-font-smoothing:antialiased;
}
.doc { max-width:1000px; margin:0 auto; padding:56px 28px 96px; }
.doc > *:not(.scroll):not(pre):not(hr):not(.lead) { max-width:70ch; }

h1, h2, h3, h4 {
  font-family:Newsreader,Georgia,"Times New Roman",serif; color:var(--ink);
  text-wrap:balance; font-weight:600; letter-spacing:-.012em;
}
h1 {
  font-size:clamp(2.1rem,4.4vw,3rem); line-height:1.08; letter-spacing:-.025em;
  margin:0 0 1.4rem;
}
h2 {
  font-size:clamp(1.4rem,2.4vw,1.8rem); line-height:1.2;
  margin:3.4rem 0 .9rem; padding-top:1.6rem; border-top:1px solid var(--rule);
}
h3 { font-size:1.14rem; margin:2.2rem 0 .6rem; }
h4 {
  font-family:"IBM Plex Sans",sans-serif; font-size:.98rem; font-weight:600;
  margin:1.6rem 0 .4rem;
}
p { margin:0 0 1.05rem; }
a { color:var(--accent); text-underline-offset:3px; }
strong { color:var(--ink); font-weight:600; }
em { color:var(--ink); }
hr { border:none; border-top:1px solid var(--rule); margin:3.2rem 0; }

/* the opening summary reads as a standfirst */
h1 + p, h1 + p + p { font-size:1.1rem; color:var(--muted); }
h1 + p strong, h1 + p + p strong { color:var(--ink); }

ul, ol { margin:0 0 1.05rem; padding-left:1.25rem; }
li { margin-bottom:.45rem; }
li > ul, li > ol { margin-top:.45rem; }

code {
  font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.85em; background:var(--raised); border:1px solid var(--rule-soft);
  border-radius:4px; padding:1px 5px; color:var(--ink);
}
pre {
  background:var(--raised); border:1px solid var(--rule); border-radius:8px;
  padding:16px 18px; overflow-x:auto; font-size:.82rem; line-height:1.7;
  margin:1.4rem 0; max-width:100%;
}
pre code { background:none; border:none; padding:0; font-size:inherit; color:var(--body); }

.scroll { overflow-x:auto; margin:1.5rem 0; border:1px solid var(--rule);
          border-radius:9px; background:var(--surface); box-shadow:var(--shadow); }
table { width:100%; border-collapse:collapse; font-size:.88rem;
        font-variant-numeric:tabular-nums; }
th, td { padding:9px 14px; text-align:right; border-bottom:1px solid var(--rule-soft);
         white-space:nowrap; }
th {
  font-family:"IBM Plex Mono",monospace; font-size:.66rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--faint); font-weight:500;
  border-bottom:1px solid var(--rule); background:var(--raised);
  position:sticky; top:0; text-align:right;
}
td:first-child, th:first-child { text-align:left; white-space:normal; min-width:12ch; }
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover td { background:var(--accent-soft); }

:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
@media (max-width:640px) {
  .doc { padding:32px 18px 64px; }
  body { font-size:15.5px; }
  h2 { margin-top:2.6rem; }
}
"""


def render(md_path: Path, out_path: Path, title: str) -> Path:
    text = md_path.read_text()
    html = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list'],
    )
    # Wide tables scroll inside their own container, never the page body.
    html = re.sub(r'<table>', '<div class="scroll"><table>', html)
    html = re.sub(r'</table>', '</table></div>', html)
    out_path.write_text(
        f'<title>{title}</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400'
        '&family=IBM+Plex+Mono:wght@400;500'
        '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">\n'
        f'<style>{CSS}</style>\n'
        f'<div class="doc">\n{html}\n</div>\n'
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default='docs/backtest.md')
    ap.add_argument('--out', default='docs/backtest.html')
    ap.add_argument('--title', default='SWM Trading Backtest')
    args = ap.parse_args()
    print('wrote', render(Path(args.md), Path(args.out), args.title))


if __name__ == '__main__':
    main()
