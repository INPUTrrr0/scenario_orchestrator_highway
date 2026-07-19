#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_demo.py — render the demo states to mp4 in v2/outputs/.

Default: one panel per state, the *uncontrolled* nominal evolution.
--compare: two panels side by side — left the nominal evolution (no
intervention), right the evolution under the minimal causal intervention
computed by directives.repair (identical when none is needed; unchanged and
labeled when repair is INFEASIBLE).

Usage:  python3 render_demo.py [--compare] [--ramp [A_MAX]] [--fps 20]
Requires PIL + ffmpeg.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import directives as dv

GRASS = (106, 153, 78)
ASPHALT = (70, 70, 74)
LANE_LINE = (230, 230, 230)
EDGE_LINE = (245, 245, 245)
HEADER_BG = (24, 24, 28)
COLORS = {  # actor id -> body color (ego green, hero-ish red, extras amber/blue)
    "0": (90, 190, 110), "1": (225, 85, 85), "2": (240, 170, 60),
    "3": (80, 130, 215), "4": (170, 110, 220),
}

VIEW = 42.0      # half-extent of the world view (m)
SCALE = 10.0     # px per meter
SIZE = int(2 * VIEW * SCALE)
GAP = 4          # px between panels in --compare


def font(sz, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, sz)
    except Exception:
        return ImageFont.load_default()


def w2p(x, y, ox, hdr):
    return (ox + SIZE / 2 + x * SCALE, hdr + SIZE / 2 - y * SCALE)


def draw_map(d, astate, ox, hdr):
    w = astate.map.lane_width
    b = w * SCALE
    cx, cy = w2p(0, 0, ox, hdr)
    d.rectangle([ox, hdr, ox + SIZE, hdr + SIZE], fill=GRASS)
    d.rectangle([cx - b, hdr, cx + b, hdr + SIZE], fill=ASPHALT)   # N-S road
    d.rectangle([ox, cy - b, ox + SIZE, cy + b], fill=ASPHALT)     # E-W road

    def dash(vert):
        s, gap = 2.0 * SCALE, 1.6 * SCALE
        p = b + gap
        while p < SIZE / 2:
            if vert:
                d.line([cx, cy + p, cx, cy + min(p + s, SIZE / 2)], fill=LANE_LINE, width=2)
                d.line([cx, cy - p, cx, cy - min(p + s, SIZE / 2)], fill=LANE_LINE, width=2)
            else:
                d.line([cx + p, cy, cx + min(p + s, SIZE / 2), cy], fill=LANE_LINE, width=2)
                d.line([cx - p, cy, cx - min(p + s, SIZE / 2), cy], fill=LANE_LINE, width=2)
            p += s + gap
    dash(True)
    dash(False)
    for sgn in (-1, 1):
        d.line([cx + sgn * b, hdr, cx + sgn * b, cy - b], fill=EDGE_LINE, width=2)
        d.line([cx + sgn * b, cy + b, cx + sgn * b, hdr + SIZE], fill=EDGE_LINE, width=2)
        d.line([ox, cy + sgn * b, cx - b, cy + sgn * b], fill=EDGE_LINE, width=2)
        d.line([cx + b, cy + sgn * b, ox + SIZE, cy + sgn * b], fill=EDGE_LINE, width=2)
    sl = [("S", (0.0, -w), (w, -w)), ("E", (w, 0.0), (w, w)),
          ("N", (0.0, w), (-w, w)), ("W", (-w, 0.0), (-w, -w))]
    for arm, a, bpt in sl:
        pa, pb = w2p(*a, ox, hdr), w2p(*bpt, ox, hdr)
        d.line([pa, pb], fill=EDGE_LINE, width=4)
        red = astate.signals.get(arm) == "red"
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        vx, vy = mx - cx, my - cy
        n = max((vx * vx + vy * vy) ** 0.5, 1e-9)
        mx += vx / n * 14
        my += vy / n * 14
        d.ellipse([mx - 6, my - 6, mx + 6, my + 6],
                  fill=(220, 40, 40) if red else (60, 200, 80), outline=(0, 0, 0))


def draw_actor(d, astate, aid, pose, ox, hdr):
    aa = astate.actors[aid]
    x, y, hdg = pose
    pts = [w2p(px, py, ox, hdr)
           for px, py in dv.rect_corners(x, y, hdg, aa.length, aa.width)]
    d.polygon(pts, fill=COLORS.get(aid, (200, 200, 200)), outline=(15, 15, 15))
    nose = w2p(*dv.rect_corners(x, y, hdg, aa.length, 0.0)[0], ox, hdr)
    d.ellipse([nose[0] - 3, nose[1] - 3, nose[0] + 3, nose[1] + 3], fill=(255, 255, 255))
    cx, cy = w2p(x, y, ox, hdr)
    d.text((cx, cy), aid, fill=(0, 0, 0), anchor="mm", font=font(14, True))


def verdict_str(res):
    return "  ".join(f"{n}:{'PASS' if e.value else 'FAIL'}"
                     for (n, _, _), e in zip(dv.DIRECTIVES, (res.d1, res.d2, res.d3)))


def first_contact(astate, evo, t_end, fps):
    ids = sorted(astate.actors)
    for k in range(int(t_end * fps) + 1):
        t = k / fps
        poses = {aid: evo.pose_at(aid, t) for aid in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                A, B = astate.actors[a], astate.actors[b]
                if dv.rects_overlap(
                        dv.rect_corners(*poses[a], A.length, A.width),
                        dv.rect_corners(*poses[b], B.length, B.width)):
                    return (t, a, b)
    return None


def draw_panel(d, astate, evo, t, hit, ox, hdr):
    draw_map(d, astate, ox, hdr)
    for aid in sorted(astate.actors):
        pose = evo.pose_at(aid, t)
        draw_actor(d, astate, aid, pose, ox, hdr)
    if hit and t >= hit[0]:
        hx, hy, _ = evo.pose_at(hit[1], t)
        px, py = w2p(hx, hy, ox, hdr)
        d.ellipse([px - 30, py - 30, px + 30, py + 30], outline=(255, 60, 60), width=5)


def encode(tmp, out, fps):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp, "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out],
        check=True)
    shutil.rmtree(tmp)


def slug_of(state):
    return re.sub(r"[^A-Za-z0-9]+", "_", state.label.split(":")[0]).strip("_")


def render_state(state, prm, outdir, fps):
    """Single panel: nominal evolution, no intervention."""
    hdr = 96
    astate = dv.recognize(state, prm)
    res = dv.evaluate_family(astate, prm)
    evo = dv.Evolution(astate, prm, {})
    t_end = min(max((res.t_star or 6.0) + 2.0, 6.5), 9.0)
    hit = first_contact(astate, evo, t_end, fps)
    tmp = tempfile.mkdtemp(prefix="frames_")
    mode = f"ramp a_max={prm.ramp:g}" if prm.ramp else "instantaneous velocity changes"
    for k in range(int(t_end * fps) + 1):
        t = k / fps
        img = Image.new("RGB", (SIZE, hdr + SIZE), HEADER_BG)
        d = ImageDraw.Draw(img)
        draw_panel(d, astate, evo, t, hit, 0, hdr)
        d.text((12, 8), state.label, fill=(240, 240, 240), font=font(19, True))
        d.text((12, 36), f"nominal evolution, no intervention   [{mode}]",
               fill=(170, 170, 170), font=font(15))
        line3 = f"t = {t:5.2f} s    {verdict_str(res)}"
        if res.d1.value and res.t_star:
            line3 += f"    planned t* = {res.t_star:.2f} s"
        if hit and t >= hit[0]:
            line3 += f"    COLLISION {hit[1]}x{hit[2]} @ {hit[0]:.2f} s"
        d.text((12, 62), line3,
               fill=(255, 90, 90) if (hit and t >= hit[0]) else (220, 220, 160),
               font=font(16, True))
        img.save(os.path.join(tmp, f"{k:04d}.png"))
    out = os.path.join(outdir, f"{slug_of(state)}_no_intervention.mp4")
    encode(tmp, out, fps)
    print(f"{out}   ({'collision %sx%s at t=%.2fs' % (hit[1], hit[2], hit[0]) if hit else 'no body contact'})")
    return out


def render_compare(state, prm, outdir, fps):
    """Two panels: nominal (left) vs post-intervention (right)."""
    hdr = 126
    astate = dv.recognize(state, prm)
    res = dv.evaluate_family(astate, prm)
    evo_l = dv.Evolution(astate, prm, {})
    controls = {}
    if res.ok:
        rr = None
        right_cap = "no intervention needed - identical evolution"
        res_r = res
    else:
        rr = dv.repair(astate, prm)
        if rr.feasible:
            for iv in rr.interventions:
                c = controls.setdefault(iv.actor, {})
                c["speed" if iv.kind == "retime" else "turn"] = iv.value
            ivs = ", ".join(
                f"{iv.kind}({iv.actor}→{iv.value:.2f} m/s)" if iv.kind == "retime"
                else f"{iv.kind}({iv.actor}→{iv.value})"
                for iv in rr.interventions)
            right_cap = f"after intervention: {ivs}   cost {rr.cost:.2f}"
        else:
            right_cap = "INFEASIBLE - no causal repair; evolution unchanged"
        res_r = rr.final
    evo_r = dv.Evolution(astate, prm, controls)
    t_end = min(max((res.t_star or 6.0), (res_r.t_star or 6.0)) + 2.0, 9.0)
    t_end = max(t_end, 6.5)
    hit_l = first_contact(astate, evo_l, t_end, fps)
    hit_r = first_contact(astate, evo_r, t_end, fps)
    width = 2 * SIZE + GAP
    ox_r = SIZE + GAP
    mode = f"ramp a_max={prm.ramp:g}" if prm.ramp else "instantaneous velocity changes"
    tmp = tempfile.mkdtemp(prefix="frames_")
    for k in range(int(t_end * fps) + 1):
        t = k / fps
        img = Image.new("RGB", (width, hdr + SIZE), HEADER_BG)
        d = ImageDraw.Draw(img)
        draw_panel(d, astate, evo_l, t, hit_l, 0, hdr)
        draw_panel(d, astate, evo_r, t, hit_r, ox_r, hdr)
        d.text((12, 8), f"{state.label}    [{mode}]    t = {t:5.2f} s",
               fill=(240, 240, 240), font=font(20, True))
        for ox, cap, r, hit, tag in (
                (0, "nominal, no intervention", res, hit_l, "L"),
                (ox_r, right_cap, res_r, hit_r, "R")):
            d.text((ox + 12, 44), cap, fill=(170, 200, 235) if ox else (170, 170, 170),
                   font=font(15))
            line = verdict_str(r)
            if r.d1.value and r.t_star:
                line += f"    t* = {r.t_star:.2f} s"
            if hit and t >= hit[0]:
                line += f"    COLLISION {hit[1]}x{hit[2]} @ {hit[0]:.2f} s"
            d.text((ox + 12, 70), line,
                   fill=(255, 90, 90) if (hit and t >= hit[0]) else (220, 220, 160),
                   font=font(16, True))
            ok = r.d1.value and r.d2.value and r.d3.value
            d.text((ox + 12, 98), "family satisfied" if ok else "family violated",
                   fill=(120, 230, 120) if ok else (250, 130, 130), font=font(15, True))
        d.line([SIZE + GAP // 2, 0, SIZE + GAP // 2, hdr + SIZE], fill=(0, 0, 0), width=GAP)
        img.save(os.path.join(tmp, f"{k:04d}.png"))
    out = os.path.join(outdir, f"{slug_of(state)}_compare.mp4")
    encode(tmp, out, fps)

    def hs(h):
        return f"collision {h[1]}x{h[2]} @ {h[0]:.2f}s" if h else "no contact"
    print(f"{out}   (left: {hs(hit_l)} | right: {hs(hit_r)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="side-by-side: nominal vs post-intervention")
    ap.add_argument("--ramp", nargs="?", const=3.0, default=None, type=float)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    prm = dv.Params(ramp=args.ramp)
    for st in dv.demo_states():
        if args.compare:
            render_compare(st, prm, args.outdir, args.fps)
        else:
            render_state(st, prm, args.outdir, args.fps)


if __name__ == "__main__":
    main()
