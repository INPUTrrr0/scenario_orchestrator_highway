#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/build_tree.py — turn a session folder into a colored, playable game tree.

For each snapshot it renders the *forward realization at that decision point* (the
re-based maneuver-script scenario rolled forward for a fixed horizon) to a looping
`snapshot_v{N}.mp4` via the render_demo pipeline, then writes `session.html`: the
tidy upside-down provenance tree of `v0/outputs/scenario_tree.html`, extended with

  * node/edge coloring by decision kind (amber = user perturbation,
    cyan = orchestrator intervention, gold = checkpoint, violet = proposal,
    red = infeasible, slate = start),
  * edge labels annotating the elapsed sim time of the collapsed NO-OP run,
  * node labels with kind, clock, the one-line delta, and a D1/D2/D3 badge,
  * a legend, with the provenance embedded so it opens directly (no fetch).

Usage:  python3 build_tree.py sessions/<session_id> [--fps 15] [--horizon 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import yaml
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
for p in (V2, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import scenario_editor as se          # noqa: E402
import render_demo as rd              # noqa: E402  (draw_map/draw_actor/w2p/encode)

KIND_COLOR = {
    "start": "#8a94a6", "intervention": "#38b2c6", "perturbation": "#e6a23c",
    "checkpoint": "#d4af37", "proposal": "#a06cd5", "infeasible": "#e05252",
}


# --------------------------------------------------------------------------- #
# Rendering: a snapshot's forward realization -> looping mp4
# --------------------------------------------------------------------------- #
class _Shim:
    pass


def _shim_astate(sc: se.Scenario, signals: dict):
    """Minimal object exposing the attributes render_demo's draw_* touch."""
    a = _Shim()
    a.map = _Shim()
    a.map.lane_width = sc.map.lane_width
    a.signals = dict(signals or {})
    a.actors = {}
    for act in sc.actors:
        o = _Shim()
        o.length, o.width, o.id = act.length, act.width, act.id
        a.actors[act.id] = o
    return a


def render_snapshot(sc_path: str, out_mp4: str, signals: dict, ego: str,
                    label: str, fps: int, horizon: float) -> None:
    sc = se.load_scenario(sc_path)
    sc.simulate()
    astate = _shim_astate(sc, signals)
    # normalize the palette to the ego-green convention
    rd.COLORS[ego] = (90, 190, 110)
    t_end = min(horizon, max(0.5, sc.period))
    n = int(t_end * fps) + 1
    hdr = 40
    tmp = tempfile.mkdtemp(prefix="v4frames_")
    for k in range(n):
        t = k / fps
        img = Image.new("RGB", (rd.SIZE, hdr + rd.SIZE), rd.HEADER_BG)
        panel = Image.new("RGB", (rd.SIZE, rd.SIZE), rd.GRASS)
        pd = ImageDraw.Draw(panel)
        rd.draw_map(pd, astate, 0, 0)
        ki = int(round(t / se.DT))
        for act in sc.actors:
            kk = max(0, min(ki, len(act.traj) - 1))
            rd.draw_actor(pd, astate, act.id, act.traj[kk], 0, 0)
        img.paste(panel, (0, hdr))
        d = ImageDraw.Draw(img)
        d.text((10, 8), label, fill=(235, 235, 235), font=rd.font(17, True))
        d.text((rd.SIZE - 120, 12), f"t={t:4.1f}s", fill=(170, 170, 170),
               font=rd.font(14))
        img.save(os.path.join(tmp, f"{k:04d}.png"))
    rd.encode(tmp, out_mp4, fps)


# --------------------------------------------------------------------------- #
# HTML game tree
# --------------------------------------------------------------------------- #
def build_html(prov: dict, out_html: str) -> None:
    data = json.dumps({"meta": prov.get("meta", {}),
                       "versions": prov.get("versions", [])})
    colors = json.dumps(KIND_COLOR)
    html = _TEMPLATE.replace("__DATA__", data).replace("__COLORS__", colors)
    with open(out_html, "w") as f:
        f.write(html)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Scenario Rollout — Game Tree</title>
<style>
  :root { --bg:#12151c; --panel:#1c212c; --edge:#4a5568; --text:#e2e8f0; --muted:#8a94a6; }
  html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  header{padding:12px 20px;border-bottom:1px solid #2a3040;}
  header h1{font-size:16px;margin:0 0 6px;font-weight:600;}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);align-items:center;}
  .legend .chip{display:inline-flex;align-items:center;gap:5px;}
  .legend .sw{width:11px;height:11px;border-radius:3px;display:inline-block;}
  #viewport{width:100%;overflow:auto;}
  #canvas{position:relative;margin:0 auto;}
  svg#edges{position:absolute;top:0;left:0;pointer-events:none;}
  svg#edges path{fill:none;stroke-width:2;}
  svg#edges text{fill:var(--muted);font-size:10.5px;}
  .node{position:absolute;width:var(--node-w);background:var(--panel);
    border:2px solid #303849;border-radius:10px;overflow:hidden;
    box-shadow:0 3px 10px rgba(0,0,0,.35);}
  .node video{display:block;width:100%;background:#000;}
  .node .bar{display:flex;align-items:center;justify-content:space-between;
    padding:4px 8px;font-size:12px;gap:6px;}
  .node .label{font-weight:600;}
  .node .kind{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;}
  .node .delta{padding:2px 8px 5px;font-size:11px;color:var(--muted);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .node .badge{display:inline-flex;gap:3px;}
  .node .dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
  .node .toggle{cursor:pointer;border:1px solid #3a4457;background:#232a38;
    color:var(--text);border-radius:6px;font-size:11px;padding:0 6px;}
  .node .toggle.leaf{visibility:hidden;}
</style></head><body>
<header>
  <h1>Scenario Rollout — Game Tree <span id="sub" style="color:var(--muted);font-weight:400;font-size:13px"></span></h1>
  <div class="legend" id="legend"></div>
</header>
<div id="viewport"><div id="canvas"><svg id="edges" xmlns="http://www.w3.org/2000/svg"></svg></div></div>
<script>
"use strict";
const DATA = __DATA__, KIND_COLOR = __COLORS__;
const NODE_W=230, NODE_H_EST=200, H_GAP=30, V_GAP=78, PAD=32;
document.documentElement.style.setProperty("--node-w", NODE_W+"px");
const canvas=document.getElementById("canvas"), edgesSvg=document.getElementById("edges");

(function legend(){
  const el=document.getElementById("legend");
  const order=["start","perturbation","intervention","checkpoint","proposal","infeasible"];
  el.innerHTML = order.map(k=>`<span class="chip"><span class="sw" style="background:${KIND_COLOR[k]}"></span>${k}</span>`).join("");
  const m=DATA.meta||{};
  document.getElementById("sub").textContent =
    `— ${m.label||""}  (ego ${m.ego}, dt ${m.dt_tick}s, ${(DATA.versions||[]).length} decision points)`;
})();

const nodes=new Map(); let roots=[];
for(const v of DATA.versions) nodes.set(v.version,{...v,children:[],collapsed:false,el:null});
for(const n of nodes.values()){
  if(n.parent!=null && nodes.has(n.parent)) nodes.get(n.parent).children.push(n);
  else roots.push(n);
}
for(const n of nodes.values()) n.children.sort((a,b)=>a.version-b.version);
roots.sort((a,b)=>a.version-b.version);

function badge(vd){
  if(!vd) return "";
  return `<span class="badge">`+["d1","d2","d3"].map(d=>
    `<span class="dot" title="${d}" style="background:${vd[d]?'#58c26a':'#e05252'}"></span>`).join("")+`</span>`;
}
function makeNodeEl(n){
  const col=KIND_COLOR[n.kind]||"#303849";
  const el=document.createElement("div"); el.className="node";
  el.style.borderColor=col;
  if(n.kind==="proposal") el.style.borderStyle="dashed";
  el.innerHTML=`
    <video muted autoplay loop playsinline preload="metadata" src="${n.file.replace('.yaml','.mp4')}"></video>
    <div class="bar">
      <span class="label">v${n.version}</span>
      <span class="kind" style="color:${col}">${n.kind}</span>
      <span>t=${(+n.sim_time).toFixed(1)}s</span>
      ${badge(n.verdict)}
      <button class="toggle ${n.children.length?'':'leaf'}">&minus;</button>
    </div>
    <div class="delta" title="${(n.delta||'').replace(/"/g,'&quot;')}">${n.delta||''}</div>`;
  el.querySelector(".toggle").addEventListener("click",()=>{
    n.collapsed=!n.collapsed;
    el.querySelector(".toggle").innerHTML=n.collapsed?"+":"&minus;"; layout();
  });
  el.querySelector("video").addEventListener("loadeddata",e=>e.target.play().catch(()=>{}));
  canvas.appendChild(el); n.el=el; return el;
}
function layout(){
  let cursorX=0; const visible=[];
  function place(n,depth){
    n.el.style.display=""; visible.push(n); n.depth=depth;
    const kids=n.collapsed?[]:n.children;
    if(n.collapsed) n.children.forEach(hide);
    if(kids.length===0){ n.x=cursorX; cursorX+=NODE_W+H_GAP; }
    else{ kids.forEach(k=>place(k,depth+1)); n.x=(kids[0].x+kids[kids.length-1].x)/2; }
  }
  function hide(n){ n.el.style.display="none"; n.children.forEach(hide); }
  roots.forEach(r=>place(r,0));
  const rowH=[]; for(const n of visible){ const h=n.el.offsetHeight||NODE_H_EST;
    rowH[n.depth]=Math.max(rowH[n.depth]||0,h); }
  const rowY=[]; let y=PAD; for(let d=0;d<rowH.length;d++){ rowY[d]=y; y+=(rowH[d]||NODE_H_EST)+V_GAP; }
  let maxX=0,maxY=0;
  for(const n of visible){ n.px=n.x+PAD; n.py=rowY[n.depth];
    n.el.style.left=n.px+"px"; n.el.style.top=n.py+"px";
    maxX=Math.max(maxX,n.px+NODE_W); maxY=Math.max(maxY,n.py+(n.el.offsetHeight||NODE_H_EST)); }
  const W=maxX+PAD,H=maxY+PAD;
  canvas.style.width=W+"px"; canvas.style.height=H+"px";
  edgesSvg.setAttribute("width",W); edgesSvg.setAttribute("height",H);
  let paths="";
  for(const n of visible){
    if(n.collapsed) continue;
    const ph=n.el.offsetHeight||NODE_H_EST;
    for(const c of n.children){
      const x1=n.px+NODE_W/2,y1=n.py+ph,x2=c.px+NODE_W/2,y2=c.py,my=(y1+y2)/2;
      const col=KIND_COLOR[c.kind]||"#4a5568";
      const dash=c.kind==="proposal"?'stroke-dasharray="6 5"':'';
      paths+=`<path d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}" stroke="${col}" ${dash}/>`;
      const dt=(+c.sim_time)-(+n.sim_time);
      if(dt>0.001) paths+=`<text x="${(x1+x2)/2+4}" y="${my}">+${dt.toFixed(1)}s</text>`;
    }
  }
  edgesSvg.innerHTML=paths;
}
const q=[...roots]; while(q.length){ const n=q.shift(); makeNodeEl(n); q.push(...n.children); }
layout();
canvas.querySelectorAll("video").forEach(v=>v.addEventListener("loadedmetadata",()=>layout(),{once:true}));
setTimeout(layout,800);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--no-render", action="store_true",
                    help="rebuild HTML only, skip mp4 rendering")
    args = ap.parse_args()

    sdir = args.session_dir if os.path.isabs(args.session_dir) \
        else os.path.join(HERE, args.session_dir)
    prov = yaml.safe_load(open(os.path.join(sdir, "provenance.yaml")))
    meta = prov.get("meta", {})
    signals = meta.get("signals", {})
    ego = str(meta.get("ego", "0"))

    if not args.no_render:
        for v in prov["versions"]:
            sc_path = os.path.join(sdir, v["file"])
            out = os.path.join(sdir, v["file"].replace(".yaml", ".mp4"))
            lbl = f"v{v['version']} · {v['kind']}"
            render_snapshot(sc_path, out, signals, ego, lbl, args.fps, args.horizon)
            print(f"rendered {os.path.basename(out)}")

    out_html = os.path.join(sdir, "session.html")
    build_html(prov, out_html)
    print(f"tree: {out_html}")


if __name__ == "__main__":
    main()
