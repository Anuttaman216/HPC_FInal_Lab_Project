#!/usr/bin/env python3
"""dashboard.py - self-contained HTML dashboard of the measured results.

A Streamlit dashboard was deliberately not used (extra dependency, not in the
Ubuntu archive). Instead this script writes results/dashboard.html: a single
static page with the summary tables, a sortable/filterable result table and
the generated graphs embedded as base64 images, so it can be opened offline
in any browser.

Usage: python3 python/dashboard.py [--results results]
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e3e2de;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#383835;--accent:#3987e5}
 img{background:#fff;border-radius:6px}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
main{max-width:1150px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.note{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:20px} .tile span{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto} table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{cursor:pointer;color:var(--muted);position:sticky;top:0;background:var(--bg)} td:nth-child(-n+3),th:nth-child(-n+3){text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}
.grid img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
input,select{font:inherit;padding:4px 8px;margin-right:8px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px}
pre{white-space:pre-wrap;font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:8px}
"""

JS = """
const tbl=document.getElementById('res');const rows=[...tbl.tBodies[0].rows];
function filt(){const q=document.getElementById('q').value.toLowerCase();const l=document.getElementById('lab').value;
 rows.forEach(r=>{const t=r.innerText.toLowerCase();r.style.display=(t.includes(q)&&(!l||r.cells[0].innerText===l))?'':'none'});}
document.getElementById('q').oninput=filt;document.getElementById('lab').onchange=filt;
[...tbl.tHead.rows[0].cells].forEach((th,i)=>{let asc=true;th.onclick=()=>{rows.sort((a,b)=>{
 const x=a.cells[i].innerText,y=b.cells[i].innerText;const nx=parseFloat(x),ny=parseFloat(y);
 const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});asc=!asc;
 rows.forEach(r=>tbl.tBodies[0].appendChild(r));};});
"""


def tile(value: str, label: str) -> str:
    return f'<div class="tile"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    try:
        df = analyze.add_scaling_metrics(analyze.load(results))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    summary = analyze.build_summary(df)

    strong = df[df["label"] == "strong"]
    tiles = []
    if not strong.empty:
        seq = strong[strong["mode"] == "sequential"]
        fastest = strong.loc[strong["time_median_s"].idxmin()]
        tiles.append(tile(strong["grid"].iloc[0], "strong-scaling grid"))
        if not seq.empty:
            tiles.append(tile(f"{seq['time_median_s'].iloc[0]:.3f} s", "sequential median time"))
        tiles.append(tile(f"{fastest['time_median_s']:.3f} s", f"fastest: {fastest['config']}"))
        if pd.notna(fastest["speedup"]):
            tiles.append(tile(f"{fastest['speedup']:.2f}x", "best speedup vs sequential"))
        tiles.append(tile(f"{strong['gflops'].max():.2f}", "peak estimated GFLOPS"))
    tiles.append(tile(str(len(df)), "measured configurations"))

    cols = [c for c in ["label", "config", "grid", "processing_units", "iterations", "time_median_s",
                        "speedup", "efficiency_pct", "gflops", "comm_fraction", "time_per_iter_ms"]
            if c in summary.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for _, r in summary.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = "n/a" if pd.isna(v) else f"{v:.4g}"
            cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    labels = "".join(f'<option value="{l}">{l}</option>' for l in sorted(summary["label"].unique()))

    imgs = []
    for png in sorted((results / "graphs").glob("*.png")):
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        imgs.append(f'<figure><img alt="{png.stem}" src="data:image/png;base64,{b64}"></figure>')

    sysinfo = (results / "system_info.txt")
    sys_txt = html.escape(sysinfo.read_text(encoding="utf-8", errors="replace")) if sysinfo.exists() else "not recorded"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CG Benchmark Dashboard</title>
<style>{CSS}</style></head><body><main>
<h1>HPCG-inspired CG benchmark dashboard</h1>
<p class="note">Educational implementation - not the official HPCG benchmark. All numbers are measured
(results/benchmark.csv); speedup and efficiency are relative to the sequential run with the same grid and tolerance.</p>
<div class="tiles">{''.join(tiles)}</div>
<h2>Graphs</h2><div class="grid">{''.join(imgs) or '<p>No graphs yet - run python/visualize.py.</p>'}</div>
<h2>All measured configurations</h2>
<p><input id="q" placeholder="filter text"><select id="lab"><option value="">all experiments</option>{labels}</select>
<span class="note">click a column header to sort</span></p>
<div class="wrap"><table id="res"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<h2>System information</h2><pre>{sys_txt}</pre>
</main><script>{JS}</script></body></html>"""
    out = results / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
