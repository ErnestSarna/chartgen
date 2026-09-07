"""Remote web UI: paste links or upload audio from any device on your VPN.

    python -m chartgen.web            # serves on http://0.0.0.0:8471
    python -m chartgen.web --port 9000

Designed to sit behind Tailscale (or another private network) on the PC with
the GPU: phones and laptops open http://<tailscale-name>:8471, paste YouTube
links or upload files, and watch the queue. Charts land in the same output
folder the desktop app uses, and every finished song can be downloaded as a
zip for a Clone Hero install on another machine.

There is deliberately no authentication: the server binds to all interfaces
but is meant to be reachable only over the private network. Do NOT port-
forward this to the open internet.

One worker thread owns the GPU, mirroring the desktop app's batch behaviour
(per-song metadata, skip-if-charted, retry pass for 403s).
"""
import argparse
import io
import json
import queue as queuemod
import threading
import time
import zipfile
from argparse import Namespace
from collections import deque
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

from . import pipeline, youtube
from .app import GRIDS, load_settings  # reuse the desktop defaults

app = Flask("chartgen")

JOBS: "queuemod.Queue[dict]" = queuemod.Queue()
LOG = deque(maxlen=400)
STATE = {"current": None, "done": [], "failed": [], "started": None}
LOCK = threading.Lock()


def log(message: str) -> None:
    with LOCK:
        LOG.append(f"{time.strftime('%H:%M:%S')}  {message}")


def outdir() -> Path:
    saved = load_settings()
    return Path(saved.get("outdir", str(Path("out").resolve())))


def base_opts() -> Namespace:
    saved = load_settings()
    return Namespace(
        audio=None, outdir=outdir(),
        subdiv=GRIDS.get(saved.get("grid"), 4),
        min_variety=0.80,
        hopos=bool(saved.get("hopos", True)),
        no_star_power=not saved.get("star_power", True),
        no_sections=not saved.get("sections", True),
        no_sustains=not saved.get("sustains", True),
        lyrics=bool(saved.get("lyrics", True)),
        target_diff=None, skip_existing=True,
        name=None, artist="Unknown", album="", genre="", year="",
    )


def worker() -> None:
    import copy

    while True:
        job = JOBS.get()
        with LOCK:
            STATE["current"] = job["label"]
            STATE["started"] = time.time()
        opts = copy.copy(base_opts())
        opts.audio = job["input"]
        try:
            result = pipeline.run(opts, progress=log)
            with LOCK:
                if result.get("skipped"):
                    STATE["done"].append(f"{Path(result['song_dir']).name} (already charted)")
                else:
                    STATE["done"].append(Path(result["song_dir"]).name)
        except Exception as error:
            log(f"FAILED: {type(error).__name__}: {error}")
            with LOCK:
                STATE["failed"].append(f"{job['label']}: {error}")
        finally:
            with LOCK:
                STATE["current"] = None


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>chartgen</title><style>
:root{
  --bg:#0d0f12; --panel:#15181d; --panel-2:#1b1f26; --line:#262b33;
  --text:#e8eaed; --muted:#8b939e; --accent:#34d17b; --accent-dim:#1f7a4b;
  --amber:#e8b84b; --red:#e2665e; --radius:12px;
}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);
  color:var(--text);margin:0;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:680px;margin:0 auto;padding:0 20px 60px}
header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.bar{max-width:680px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:12px}
.logo{width:26px;height:26px;flex:none}
h1{font-size:17px;font-weight:650;margin:0;letter-spacing:-.01em}
h1 small{color:var(--muted);font-weight:400;margin-left:8px;font-size:13px}
.pill{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;
  padding:5px 12px;border-radius:99px;border:1px solid var(--line);color:var(--muted);
  background:var(--panel);white-space:nowrap}
.pill.busy{color:var(--amber);border-color:color-mix(in srgb,var(--amber) 35%,var(--line))}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex:none}
.busy .dot{background:var(--amber);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
section{margin-top:28px}
.label{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
textarea{width:100%;min-height:96px;resize:vertical;background:var(--panel-2);
  color:var(--text);border:1px solid var(--line);border-radius:8px;padding:12px;
  font:inherit;font-size:15px}
textarea::placeholder{color:var(--muted)}
textarea:focus,button:focus-visible{outline:2px solid var(--accent-dim);outline-offset:1px;border-color:transparent}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{font:inherit;font-size:14px;font-weight:600;border:none;border-radius:8px;
  padding:10px 18px;cursor:pointer;background:var(--accent);color:#08130c;
  transition:filter .15s}
button:hover{filter:brightness(1.1)}
button:disabled{opacity:.45;cursor:default;filter:none}
button.ghost{background:var(--panel-2);color:var(--text);border:1px solid var(--line)}
.or{color:var(--muted);font-size:13px}
.drop{margin-top:12px;border:1.5px dashed var(--line);border-radius:8px;padding:18px;
  text-align:center;color:var(--muted);font-size:14px;cursor:pointer;transition:border-color .15s,background .15s}
.drop:hover,.drop.over{border-color:var(--accent-dim);background:var(--panel-2);color:var(--text)}
.now{display:flex;align-items:center;gap:12px;padding:14px 16px}
.eq{display:flex;gap:3px;align-items:flex-end;height:18px;flex:none}
.eq span{width:4px;border-radius:2px;background:var(--accent);animation:eq 1s ease-in-out infinite}
.eq span:nth-child(2){animation-delay:.2s}.eq span:nth-child(3){animation-delay:.4s}
@keyframes eq{0%,100%{height:6px}50%{height:18px}}
.now .what{min-width:0}
.now .name{font-weight:600;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.now .sub{color:var(--muted);font-size:12.5px}
details{margin-top:10px}
summary{cursor:pointer;color:var(--muted);font-size:13px;user-select:none}
pre{background:var(--panel-2);border:1px solid var(--line);margin:10px 0 0;padding:12px;
  border-radius:8px;font-size:12px;line-height:1.6;max-height:240px;overflow:auto;
  white-space:pre-wrap;color:var(--muted);font-family:ui-monospace,Consolas,monospace}
.song{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}
.song:last-child{border-bottom:none}
.song .name{flex:1;min-width:0;font-size:14.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.song .note{color:var(--muted);font-size:12.5px}
.song.fail .name{color:var(--red);white-space:normal;font-size:13px}
a.dl{flex:none;display:flex;align-items:center;gap:6px;text-decoration:none;font-size:13px;
  font-weight:600;color:var(--accent);border:1px solid var(--line);border-radius:8px;
  padding:7px 12px;transition:background .15s}
a.dl:hover{background:var(--panel-2)}
.empty{padding:22px 16px;color:var(--muted);font-size:14px;text-align:center}
.cardlist{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}
.count{color:var(--muted);font-weight:400}
</style></head><body>
<header><div class="bar">
<svg class="logo" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  <rect x="2" y="3" width="20" height="18" rx="4" fill="#1f7a4b"/>
  <rect x="5.5" y="7" width="3" height="10" rx="1.5" fill="#34d17b"/>
  <rect x="10.5" y="10" width="3" height="7" rx="1.5" fill="#e8b84b"/>
  <rect x="15.5" y="6" width="3" height="11" rx="1.5" fill="#e2665e"/>
</svg>
<h1>chartgen<small>Clone Hero chart generator</small></h1>
<div class="pill" id="pill"><span class="dot"></span><span id="pilltext">Idle</span></div>
</div></header>
<div class="wrap">

<section>
  <div class="label">Add songs</div>
  <div class="card">
    <form id="f">
      <textarea id="links" placeholder="Paste YouTube links or a playlist URL — one per line"></textarea>
      <div class="row">
        <button type="submit" id="chartbtn">Chart them</button>
        <span class="or">or drop audio files below</span>
      </div>
    </form>
    <div class="drop" id="drop">Click to choose audio files — or drag them here</div>
    <input type="file" id="file" accept="audio/*" multiple hidden>
  </div>
</section>

<section id="activity" hidden>
  <div class="label">Now charting</div>
  <div class="card now">
    <div class="eq" aria-hidden="true"><span></span><span></span><span></span></div>
    <div class="what">
      <div class="name" id="curname"></div>
      <div class="sub" id="cursub"></div>
    </div>
  </div>
  <details><summary>Show log</summary><pre id="log"></pre></details>
</section>

<section>
  <div class="label">Finished <span class="count" id="donecount"></span></div>
  <div class="cardlist" id="done"><div class="empty">Nothing charted yet — paste a link above.</div></div>
</section>

<section id="failsec" hidden>
  <div class="label" style="color:var(--red)">Failed</div>
  <div class="cardlist" id="failed"></div>
</section>

</div>
<script>
const $=id=>document.getElementById(id);
const f=$('f'), drop=$('drop'), file=$('file'), btn=$('chartbtn');

f.onsubmit=async e=>{e.preventDefault();
  const links=$('links').value.trim(); if(!links)return;
  btn.disabled=true;
  try{
    const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({links:links.split(/\\s+/).filter(Boolean)})});
    if(r.ok) $('links').value='';
    else alert((await r.json()).error||'Could not queue those links.');
  }finally{btn.disabled=false;}
  refresh();};

drop.onclick=()=>file.click();
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>sendFiles(e.dataTransfer.files));
file.onchange=()=>sendFiles(file.files);
async function sendFiles(list){
  for(const item of list){const fd=new FormData();fd.append('file',item);
    await fetch('/upload',{method:'POST',body:fd});}
  file.value='';refresh();}

function songRow(name){
  const clean=name.replace(' (already charted)','');
  const row=document.createElement('div');row.className='song';
  const n=document.createElement('div');n.className='name';n.textContent=clean;
  if(name!==clean){const s=document.createElement('span');s.className='note';
    s.textContent=' already charted';n.appendChild(s);}
  const a=document.createElement('a');a.className='dl';
  a.href='/songs/'+encodeURIComponent(clean)+'.zip';
  a.innerHTML='&#8681; zip';a.title='Download for another Clone Hero install';
  row.append(n,a);return row;}

async function refresh(){
  let s;try{s=await(await fetch('/status')).json();}catch(e){
    $('pilltext').textContent='PC unreachable';$('pill').classList.remove('busy');return;}
  const busy=!!s.current||s.queued>0;
  $('pill').classList.toggle('busy',busy);
  $('pilltext').textContent=s.current?('Working · '+s.queued+' queued'):(s.queued?s.queued+' queued':'Idle');
  $('activity').hidden=!busy;
  if(s.current){$('curname').textContent=s.current;
    $('cursub').textContent=s.queued?s.queued+' more in queue':'last one in the queue';}
  $('log').textContent=s.log.join('\\n');
  const done=$('done');done.innerHTML='';
  $('donecount').textContent=s.done.length?('· '+s.done.length):'';
  if(!s.done.length){done.innerHTML='<div class="empty">Nothing charted yet — paste a link above.</div>';}
  else for(const d of s.done.slice().reverse()) done.appendChild(songRow(d));
  $('failsec').hidden=!s.failed.length;
  const failed=$('failed');failed.innerHTML='';
  for(const d of s.failed.slice().reverse()){const row=document.createElement('div');
    row.className='song fail';const n=document.createElement('div');
    n.className='name';n.textContent=d;row.appendChild(n);failed.appendChild(row);}
}
setInterval(refresh,2500);refresh();
</script></body></html>"""


@app.get("/")
def index() -> Response:
    return Response(PAGE, mimetype="text/html")


@app.post("/jobs")
def add_jobs():
    links = (request.get_json(silent=True) or {}).get("links", [])
    links = [l.strip() for l in links if l.strip()]
    bad = [l for l in links if not youtube.is_youtube_url(l)]
    if bad:
        return jsonify({"error": f"not YouTube links: {bad}"}), 400
    try:
        expanded = youtube.expand_inputs(links, log)
    except ValueError as error:
        return jsonify({"error": str(error)}), 502
    for url in expanded:
        JOBS.put({"input": url, "label": url})
    log(f"queued {len(expanded)} link(s)")
    return jsonify({"queued": len(expanded)})


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file"}), 400
    dest = outdir() / "_uploads"
    dest.mkdir(parents=True, exist_ok=True)
    safe = Path(file.filename).name
    path = dest / safe
    file.save(path)
    JOBS.put({"input": path, "label": safe})
    log(f"queued upload {safe}")
    return jsonify({"queued": 1})


@app.get("/status")
def status():
    with LOCK:
        return jsonify({
            "current": STATE["current"], "queued": JOBS.qsize(),
            "done": list(STATE["done"]), "failed": list(STATE["failed"]),
            "log": list(LOG)[-60:],
        })


@app.get("/songs/<name>.zip")
def song_zip(name: str):
    folder = (outdir() / name).resolve()
    if not str(folder).startswith(str(outdir().resolve())) or not folder.is_dir():
        abort(404)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in folder.iterdir():
            if f.is_file():
                zf.write(f, arcname=f"{name}/{f.name}")
    buffer.seek(0)
    return send_file(buffer, download_name=f"{name}.zip",
                     mimetype="application/zip")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8471)
    args = ap.parse_args()

    threading.Thread(target=worker, daemon=True).start()
    log(f"chartgen web up — output folder: {outdir()}")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
