#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maps.py — road map definition and pygame rendering.

Two kinds:
  * intersection — 4-way crossroads (default), sized by lane_width × arm_length
  * straight     — multi-lane strip along +y (north), sized by num_lanes ×
                   lane_width × length

World frame: meters, origin at road center, x=East, y=North.
Lanes on a straight map are northbound; lane_center_x(0) is the leftmost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pygame

WorldToScreen = Callable[[float, float], Tuple[int, int]]
ScreenToWorld = Callable[[float, float], Tuple[float, float]]

# default draw palette (callers may override)
C_ROAD = (60, 60, 66)
C_LINE = (220, 210, 120)
C_EDGE = (200, 200, 200)
C_RULER_BG = (28, 30, 36)
C_RULER_FG = (170, 176, 188)
C_RULER_MAJOR = (230, 232, 238)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class MapConfig:
    """Parametric road layout — no mesh, just numbers the renderer turns into
    asphalt / lane markings."""
    lane_width: float = 3.5
    arm_length: float = 60.0       # intersection arm length from center
    kind: str = "intersection"     # "intersection" | "straight"
    num_lanes: int = 2             # same-direction lanes (straight maps)
    length: float = 120.0          # total road length for straight maps

    def lane_center_x(self, lane_index: int) -> float:
        """World x of a northbound lane center (0 = leftmost)."""
        n = max(1, self.num_lanes)
        i = int(clamp(float(lane_index), 0, n - 1))
        half_w = n * self.lane_width / 2.0
        return -half_w + self.lane_width / 2.0 + i * self.lane_width

    def half_width(self) -> float:
        if self.kind == "straight":
            return max(1, self.num_lanes) * self.lane_width / 2.0
        return self.lane_width

    def half_length(self) -> float:
        if self.kind == "straight":
            return self.length / 2.0
        return self.arm_length

    def to_dict(self) -> dict:
        m: dict = {"kind": self.kind, "lane_width": self.lane_width}
        if self.kind == "straight":
            m["num_lanes"] = self.num_lanes
            m["length"] = self.length
        else:
            m["arm_length"] = self.arm_length
        return m


def clone_map(m: MapConfig) -> MapConfig:
    return MapConfig(lane_width=m.lane_width, arm_length=m.arm_length,
                     kind=m.kind, num_lanes=m.num_lanes, length=m.length)


def map_from_dict(mp: dict | None) -> MapConfig:
    """Build a MapConfig from a YAML `map:` block (or empty dict)."""
    mp = mp or {}
    kind = str(mp.get("kind", "intersection")).lower()
    if kind not in ("intersection", "straight"):
        kind = "intersection"
    return MapConfig(
        lane_width=float(mp.get("lane_width", 3.5)),
        arm_length=float(mp.get("arm_length", 60.0)),
        kind=kind,
        num_lanes=int(mp.get("num_lanes", 3 if kind == "straight" else 2)),
        length=float(mp.get("length", 120.0)),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def draw_map(surface: pygame.Surface, m: MapConfig, w2s: WorldToScreen,
             road=C_ROAD, line=C_LINE, edge=C_EDGE,
             extend_y: float = 0.0) -> None:
    """Draw the road onto `surface`. `w2s(wx, wy) -> (sx, sy)` converts world
    meters to screen pixels. `extend_y` lengthens a straight road beyond
    ±length/2 (useful when a following camera needs margin)."""
    if m.kind == "straight":
        _draw_straight(surface, m, w2s, road, line, edge, extend_y)
    else:
        _draw_intersection(surface, m, w2s, road, line, edge)


def _draw_straight(surface, m: MapConfig, w2s, road, line, edge,
                   extend_y: float) -> None:
    lw = m.lane_width
    n = max(1, m.num_lanes)
    half_w = n * lw / 2.0
    half_l = m.length / 2.0
    y0, y1 = -half_l - extend_y, half_l + extend_y
    p0 = w2s(-half_w, y1)
    p1 = w2s(half_w, y0)
    pygame.draw.rect(surface, road,
                     pygame.Rect(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]))
    pygame.draw.line(surface, edge, w2s(-half_w, y0), w2s(-half_w, y1), 2)
    pygame.draw.line(surface, edge, w2s(half_w, y0), w2s(half_w, y1), 2)
    dash = 3.0
    for i in range(1, n):
        x = -half_w + i * lw
        d = y0
        while d < y1:
            pygame.draw.line(surface, line, w2s(x, d),
                             w2s(x, min(d + dash, y1)), 2)
            d += dash * 2


def _draw_intersection(surface, m: MapConfig, w2s, road, line, edge) -> None:
    arm = m.arm_length
    half = m.lane_width
    for (a, b) in (((-arm, half), (arm, -half)), ((-half, arm), (half, -arm))):
        x0, y0 = w2s(*a)
        x1, y1 = w2s(*b)
        pygame.draw.rect(surface, road,
                         pygame.Rect(x0, y0, x1 - x0, y1 - y0))
    dash, d = 3.0, -arm
    while d < arm:
        if abs(d) > half:
            pygame.draw.line(surface, line, w2s(d, 0),
                             w2s(min(d + dash, arm), 0), 2)
            pygame.draw.line(surface, line, w2s(0, d),
                             w2s(0, min(d + dash, arm)), 2)
        d += dash * 2
    for sx, sy, ex, ey in [(-half, -half, 0, -half), (0, half, half, half),
                           (-half, half, -half, 0), (half, -half, half, 0)]:
        pygame.draw.line(surface, edge, w2s(sx, sy), w2s(ex, ey), 2)


# --------------------------------------------------------------------------- #
# Distance rulers (screen-space bands labelled in world meters)
# --------------------------------------------------------------------------- #
def _nice_step(span_m: float, target_ticks: int = 10) -> float:
    """Pick a 1/2/5×10^k meter step so ~target_ticks fit in `span_m`."""
    if span_m <= 1e-6:
        return 1.0
    raw = span_m / max(target_ticks, 1)
    exp = math.floor(math.log10(raw))
    base = 10.0 ** exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        if mult * base >= raw:
            return mult * base
    return 10.0 * base


def draw_rulers(surface: pygame.Surface,
                w2s: WorldToScreen,
                s2w: ScreenToWorld,
                view: pygame.Rect,
                font: Optional[pygame.font.Font] = None,
                band: int = 30,
                bg=C_RULER_BG,
                fg=C_RULER_FG,
                major=C_RULER_MAJOR) -> None:
    """Draw meter rulers along the top (x / East) and left (y / North) of
    `view`. Tick labels are world coordinates in meters."""
    if font is None:
        font = pygame.font.SysFont("consolas,menlo,monospace", 11)

    inner = pygame.Rect(view.x + band, view.y + band,
                        max(1, view.width - band), max(1, view.height - band))
    wx0, wy_top = s2w(inner.left, inner.top)
    wx1, wy_bot = s2w(inner.right, inner.bottom)
    x_lo, x_hi = (wx0, wx1) if wx0 <= wx1 else (wx1, wx0)
    y_lo, y_hi = (wy_bot, wy_top) if wy_bot <= wy_top else (wy_top, wy_bot)

    step = _nice_step(max(x_hi - x_lo, y_hi - y_lo))
    minor = step / 5.0 if step >= 5.0 else step / 2.0

    pygame.draw.rect(surface, bg, pygame.Rect(view.x, view.y, view.width, band))
    pygame.draw.rect(surface, bg, pygame.Rect(view.x, view.y, band, view.height))
    pygame.draw.rect(surface, bg, pygame.Rect(view.x, view.y, band, band))
    pygame.draw.line(surface, fg, (view.x, view.y + band),
                     (view.right, view.y + band), 1)
    pygame.draw.line(surface, fg, (view.x + band, view.y),
                     (view.x + band, view.bottom), 1)

    unit = font.render("m", True, major)
    surface.blit(unit, (view.x + (band - unit.get_width()) // 2,
                        view.y + (band - unit.get_height()) // 2))

    def _is_major(v: float) -> bool:
        return abs(round(v / step) * step - v) < step * 1e-6

    x = math.floor(x_lo / minor) * minor
    while x <= x_hi + 1e-9:
        sx, _ = w2s(x, (y_lo + y_hi) * 0.5)
        if inner.left <= sx <= inner.right:
            maj = _is_major(x)
            tick_h = 12 if maj else 6
            pygame.draw.line(surface, major if maj else fg,
                             (sx, view.y + band), (sx, view.y + band - tick_h), 1)
            if maj:
                lbl = font.render(f"{x:g}", True, major)
                lx = sx - lbl.get_width() // 2
                lx = max(view.x + band + 1,
                         min(lx, view.right - lbl.get_width() - 1))
                surface.blit(lbl, (lx, view.y + 4))
        x += minor

    y = math.floor(y_lo / minor) * minor
    while y <= y_hi + 1e-9:
        _, sy = w2s((x_lo + x_hi) * 0.5, y)
        if inner.top <= sy <= inner.bottom:
            maj = _is_major(y)
            tick_w = 12 if maj else 6
            pygame.draw.line(surface, major if maj else fg,
                             (view.x + band, sy),
                             (view.x + band - tick_w, sy), 1)
            if maj:
                lbl = font.render(f"{y:g}", True, major)
                ly = sy - lbl.get_height() // 2
                ly = max(view.y + band + 1,
                         min(ly, view.bottom - lbl.get_height() - 1))
                surface.blit(lbl, (view.x + 3, ly))
        y += minor
