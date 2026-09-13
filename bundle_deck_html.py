"""Bundles the adxppt per-slide output into one self-contained navigable HTML file.

The converter emits one file per slide plus an index. This merges them into a
single document with keyboard / click / button navigation and auto-scaling to
the viewport. Assumes no external assets (the deck is pure shapes and text).
"""

import html
import re
from pathlib import Path

SRC = Path("docs/deck_html")
OUT = Path("docs/email_intent_briefing.html")
DECK_TITLE = "Email Intent Classification — Decision Briefing"

slide_files = sorted((SRC / "slides").glob("slide*.html"))
if not slide_files:
    raise SystemExit(f"no slides found under {SRC/'slides'} — run the converter first")

index_html = (SRC / "index.html").read_text(encoding="utf-8")
titles = {
    m.group(1): re.sub(r"<[^>]+>", "", m.group(2)).strip()
    for m in re.finditer(r'href="slides/(slide\d+\.html)"[^>]*>(.*?)</a>', index_html, re.S)
}

BODY_RE = re.compile(
    r'<div class="slide-wrap"><div class="slide">(.*?)</div></div>\s*</body>', re.S
)

slides, names = [], []
for f in slide_files:
    text = f.read_text(encoding="utf-8")
    m = BODY_RE.search(text)
    if not m:
        raise SystemExit(f"could not locate slide body in {f}")
    slides.append(m.group(1).strip())
    raw = titles.get(f.name, "").strip() or f"Slide {len(slides)}"
    raw = re.sub(r"^Slide\s+\d+\s*[:.–-]\s*", "", raw)
    names.append(re.sub(r"\s+", " ", raw)[:70])

if len(slides) > 1:
    # Slide 1 concatenates the two title-placeholder paragraphs into one label.
    names[0] = names[0].split("Build our own")[0].strip() or names[0]

sections = "\n".join(
    f'<section class="slide" data-i="{i}">{body}</section>'
    for i, body in enumerate(slides)
)
dots = "\n".join(
    f'<button class="dot" data-go="{i}" title="{html.escape(n)}">{i+1}</button>'
    for i, n in enumerate(names)
)

doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(DECK_TITLE)}</title>
<style>
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; height:100%; background:#1b1f27;
                font-family:Arial, Helvetica, sans-serif; overflow:hidden; }}
  #stage {{ position:absolute; inset:0 0 56px 0; display:flex;
            align-items:center; justify-content:center; overflow:hidden; }}
  #canvas {{ position:relative; width:1280px; height:720px; flex:0 0 auto;
             transform-origin:center center; }}
  section.slide {{ position:absolute; inset:0; width:1280px; height:720px;
                   background:#fff; box-shadow:0 6px 30px rgba(0,0,0,.45);
                   display:none; }}
  section.slide.active {{ display:block; }}
  .shape {{ font-size:16px; color:#222; }}

  #bar {{ position:absolute; left:0; right:0; bottom:0; height:56px;
          display:flex; align-items:center; gap:14px; padding:0 18px;
          background:#11141a; border-top:1px solid #2b313c; color:#c8cfdb;
          font-size:13px; user-select:none; }}
  button {{ font:inherit; color:#c8cfdb; background:#222833; border:1px solid #39414f;
            border-radius:6px; padding:7px 14px; cursor:pointer; }}
  button:hover:not(:disabled) {{ background:#2e3646; color:#fff; }}
  button:disabled {{ opacity:.35; cursor:default; }}
  #dots {{ display:flex; gap:5px; margin-left:auto; flex-wrap:nowrap; }}
  .dot {{ padding:5px 9px; font-size:12px; }}
  .dot.on {{ background:#576CBC; border-color:#576CBC; color:#fff; font-weight:bold; }}
  #label {{ min-width:330px; color:#8d97a8; }}
  #label b {{ color:#e7ecf4; font-weight:600; }}
  #zones {{ position:absolute; inset:0; display:flex; }}
  #zones div {{ flex:1; cursor:pointer; }}
  #zones .back {{ flex:0 0 22%; }}
</style>
</head>
<body>
<div id="stage">
  <div id="canvas">
{sections}
    <div id="zones"><div class="back" title="Previous"></div><div title="Next"></div></div>
  </div>
</div>
<div id="bar">
  <button id="prev">&larr; Back</button>
  <button id="next">Next &rarr;</button>
  <span id="label"></span>
  <span id="dots">{dots}</span>
</div>
<script>
const NAMES = {names!r};
const slides = [...document.querySelectorAll('section.slide')];
const dots = [...document.querySelectorAll('.dot')];
let cur = 0;

function show(i) {{
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, n) => s.classList.toggle('active', n === cur));
  dots.forEach((d, n) => d.classList.toggle('on', n === cur));
  document.getElementById('label').innerHTML =
    '<b>' + (cur + 1) + ' / ' + slides.length + '</b> &nbsp; ' + NAMES[cur];
  document.getElementById('prev').disabled = cur === 0;
  document.getElementById('next').disabled = cur === slides.length - 1;
  if (location.hash !== '#' + (cur + 1)) history.replaceState(null, '', '#' + (cur + 1));
}}

function fit() {{
  const st = document.getElementById('stage');
  const k = Math.min(st.clientWidth / 1280, st.clientHeight / 720);
  document.getElementById('canvas').style.transform = 'scale(' + k + ')';
}}

document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
dots.forEach(d => d.onclick = () => show(+d.dataset.go));
const z = document.querySelectorAll('#zones div');
z[0].onclick = () => show(cur - 1);
z[1].onclick = () => show(cur + 1);

addEventListener('keydown', e => {{
  if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(e.key)) {{ show(cur + 1); e.preventDefault(); }}
  else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) {{ show(cur - 1); e.preventDefault(); }}
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key === 'f') document.documentElement.requestFullscreen?.();
}});

addEventListener('resize', fit);
fit();
show(Math.max(0, (parseInt(location.hash.slice(1), 10) || 1) - 1));
</script>
</body>
</html>
"""

OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT}  ({len(slides)} slides, {OUT.stat().st_size/1024:.0f} KB, self-contained)")
