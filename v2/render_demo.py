#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_demo.py — render each demo state's *uncontrolled* nominal evolution
(no intervention applied) to an mp4 in v2/outputs/.

Usage:  python3 render_demo.py [--ramp [A_MAX]] [--fps 20] [--outdir outputs]
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
COLORS = {  # actor id -> body color (ego green, hero-ish red, extras blue/amber)
    "0": (90, 190, 110), "1": (225, 85, 85), "2": (240, 170, 60),
    "3": (80, 130, 215), "4": (170, 110, 220),
}

VIEW = 42.0      # half-extent of the world view (m)
SCALE = 10.0     # px per meter
HDR = 96         # header strip height (px)
SIZE = int(2 * VIEW * SCALE)


def font(sz, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, sz)
    except Exception:
        return ImageFont.load_default()


def w2p(x, y):
    return (SIZE / 2 + x * SCALE, HDR + SIZE / 2 - y * SCALE)


def draw_map(d: ImageDraw.ImageDraw, astate):
    w = astate.map.lane_width
    b = w * SCALE
    cx, cy = w2p(0, 0)
    d.rectangle([0, HDR, SIZE, HDR + SIZE], fill=GRASS)
    d.rectangle([cx - b, HDR, cx + b, HDR + SIZE], fill=ASPHALT)   # N-S road
    d.rectangle([0, cy - b, SIZE, cy + b], fill=ASPHALT)           # E-W road
    # dashed center lines (outside the box) + solid edges
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
        d.line([cx + sgn * b, HDR, cx + sgn * b, cy - b], fill=EDGE_LINE, width=2)
        d.line([cx + sgn * b, cy + b, cx + sgn * b, HDR + SIZE], fill=EDGE_LINE, width=2)
        d.line([0, cy + sgn * b, cx - b, cy + sgn * b], fill=EDGE_LINE, width=2)
        d.line([cx + b, cy + sgn * b, SIZE, cy + sgn * b], fill=EDGE_LINE, width=2)
    # stop lines (inbound lane halves) + signal dots per arm
    sl = [("S", (0.0, -w), (w, -w)), ("E", (w, 0.0), (w, w)),
          ("N", (0.0, w), (-w, w)), ("W", (-w, 0.0), (-w, -w))]
    for arm, a, bpt in sl:
        pa, pb = w2p(*a), w2p(*bpt)
        d.line([pa, pb], fill=EDGE_LINE, width=4)
        red = astate.signals.get(arm) == "red"
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        # offset the dot outward from the box center
        ox = mx - w2p(0, 0)[0]
        oy = my - w2p(0, 0)[1]
        n = max((ox * ox + oy * oy) ** 0.5, 1e-9)
        mx += ox / n * 14
        my += oy / n * 14
        d.ellipse([mx - 6, my - 6, mx + 6, my + 6],
                  fill=(220, 40, 40) if red else (60, 200, 80),
                  outline=(0, 0, 0))


def draw_actor(d, astate, aid, pose):
    aa = astate.actors[aid]
    x, y, hdg = pose
    corners = dv.rect_corners(x, y, hdg, aa.length, aa.width)
    pts = [w2p(px, py) for px, py in corners]
    col = COLORS.get(aid, (200, 200, 200))
    d.polygon(pts, fill=col, outline=(15, 15, 15))
    nose = w2p(*dv.rect_corners(x, y, hdg, aa.length * 1.0, 0.0)[0])
    d.ellipse([nose[0] - 3, nose[1] - 3, nose[0] + 3, nose[1] + 3], fill=(255, 255, 255))
    cx, cy = w2p(x, y)
    d.text((cx, cy), aid, fill=(0, 0, 0), anchor="mm", font=font(14, True))


def render_state(state, prm, outdir, fps):
    astate = dv.recognize(state, prm)
    res = dv.evaluate_family(astate, prm)   # verdicts of the uncontrolled state
    evo = dv.Evolution(astate, prm, {})     # nominal evolution, NO intervention
    t_end = min(max((res.t_star or 6.0) + 2.0, 6.5), 9.0)
    verdict = "   ".join(
        f"{n}:{'PASS' if e.value else 'FAIL'}"
        for (n, _, _), e in zip(dv.DIRECTIVES, (res.d1, res.d2, res.d3)))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", state.label.split(":")[0]).strip("_")
    tmp = tempfile.mkdtemp(prefix="frames_")
    first_hit = None
    n_frames = int(t_end * fps) + 1
    ids = sorted(astate.actors)
    for k in range(n_frames):
        t = k / fps
        img = Image.new("RGB", (SIZE, HDR + SIZE), HEADER_BG)
        d = ImageDraw.Draw(img)
        draw_map(d, astate)
        poses = {aid: evo.pose_at(aid, t) for aid in ids}
        # any body overlap this frame?
        if first_hit is None:
            for i, a in enumerate(ids):
                for b_ in ids[i + 1:]:
                    A, B = astate.actors[a], astate.actors[b_]
                    ra = dv.rect_corners(*poses[a], A.length, A.width)
                    rb = dv.rect_corners(*poses[b_], B.length, B.width)
                    if dv.rects_overlap(ra, rb):
                        first_hit = (t, a, b_)
                        break
                if first_hit:
                    break
        for aid in ids:
            draw_actor(d, astate, aid, poses[aid])
        if first_hit and t >= first_hit[0]:
            hx, hy, _ = poses[first_hit[1]]
            px, py = w2p(hx, hy)
            d.ellipse([px - 30, py - 30, px + 30, py + 30], outline=(255, 60, 60), width=5)
        # header
        d.text((12, 8), state.label, fill=(240, 240, 240), font=font(19, True))
        mode = f"ramp a_max={prm.ramp:g}" if prm.ramp else "instantaneous velocity changes"
        d.text((12, 36), f"nominal evolution, no intervention   [{mode}]",
               fill=(170, 170, 170), font=font(15))
        line3 = f"t = {t:5.2f} s    {verdict}"
        if res.d1.value and res.t_star:
            line3 += f"    planned t* = {res.t_star:.2f} s"
        if first_hit and t >= first_hit[0]:
            line3 += f"    COLLISION {first_hit[1]}x{first_hit[2]} @ {first_hit[0]:.2f} s"
        d.text((12, 62), line3,
               fill=(255, 90, 90) if (first_hit and t >= first_hit[0]) else (220, 220, 160),
               font=font(16, True))
        img.save(os.path.join(tmp, f"{k:04d}.png"))
    out = os.path.join(outdir, f"{slug}_no_intervention.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp, "%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out],
        check=True)
    shutil.rmtree(tmp)
    hit = f"collision {first_hit[1]}x{first_hit[2]} at t={first_hit[0]:.2f}s" \
        if first_hit else "no body contact"
    print(f"{out}   ({n_frames} frames, {hit})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramp", nargs="?", const=3.0, default=None, type=float)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    prm = dv.Params(ramp=args.ramp)
    for st in dv.demo_states():
        render_state(st, prm, args.outdir, args.fps)


if __name__ == "__main__":
    main()
