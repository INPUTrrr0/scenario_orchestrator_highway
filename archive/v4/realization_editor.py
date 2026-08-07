#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/realization_editor.py — the v0 scenario editor, refactored into an embeddable
pane.

Same UI and behavior as v0/scenario_editor.py's `run_gui` (top bar with Play /
Reset / +Actor / -Actor and an editable T field; a bird's-eye canvas with
click-select and draggable spawn/rotation handles; a timing-curve subwindow with
prev/next, a cycling `type:` button, editable fields, and draggable velocity-curve
endpoints) — but drawn into a sub-region at a caller-chosen offset and driven by an
external event loop, instead of owning the whole window and loop. It edits a v2
`scenario_editor.Scenario` in place (the class the v4 kernel uses), calling
`scenario.simulate()` after each edit.

It knows nothing about the session/game-tree; it just edits the loaded realization
and tracks `dirty`. Persistence (checkpoint / commit as a perturbation) is the
session shell's job.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "v2")
if V2 not in sys.path:
    sys.path.insert(0, V2)

import scenario_editor as se          # noqa: E402  (Scenario/Actor/Maneuver used by the kernel)

Pose = Tuple[float, float, float]
LONGITUDINAL_TYPES = {"go_straight", "accelerate", "decelerate"}
MANEUVER_TYPE_CYCLE = ["go_straight", "turn_left", "turn_right",
                       "accelerate", "decelerate", "stop"]

C_GRASS = (32, 44, 34)
C_ROAD = (60, 60, 66)
C_LINE = (220, 210, 120)
C_EDGE = (200, 200, 200)
C_BAR = (24, 26, 32)
C_BTN = (54, 58, 70)
C_BTN_HL = (80, 120, 200)
C_TEXT = (230, 230, 235)
C_PANEL = (30, 32, 40)
C_SEL = (255, 220, 40)
C_AXIS = (150, 150, 160)
C_CURVE = (120, 200, 255)

ADD_PRESETS = [(1.75, -58, 90), (58, 1.75, 180), (-58, -1.75, 0), (-1.75, 58, 270)]
ADD_PALETTE = [(90, 190, 110), (210, 70, 60), (60, 120, 210),
               (200, 160, 60), (160, 90, 200), (80, 200, 200)]


class RealizationEditor:
    def __init__(self, scenario: se.Scenario, w: int = 1000, h: int = 820):
        pygame.font.init()
        self.W, self.H = w, h
        self.TOPBAR_H = 56
        self.SUBWIN_H = 240
        self.CANVAS_H = h - self.TOPBAR_H - self.SUBWIN_H
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_sm = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)
        self.canvas_cx = w // 2
        self.canvas_cy = self.TOPBAR_H + self.CANVAS_H // 2

        # top-bar rects (local coords)
        self.btn_play = pygame.Rect(w // 2 - 55, 10, 110, 36)
        self.btn_reset = pygame.Rect(w // 2 - 190, 10, 110, 36)
        self.btn_add = pygame.Rect(w - 240, 10, 105, 36)
        self.btn_del = pygame.Rect(w - 130, 10, 105, 36)
        self.time_field = pygame.Rect(52, 14, 84, 28)

        self._s: Optional[pygame.Surface] = None
        self.set_scenario(scenario)

    # ------------------------------------------------------------------ #
    def set_scenario(self, scenario: se.Scenario) -> None:
        self.scenario = scenario
        self.scenario.simulate()
        self.playing = False
        self.T = 0.0
        self.selected: Optional[int] = None
        self.man_index = 0
        self.focus_field: Optional[str] = None
        self.edit_buffer = ""
        self.dragging: Optional[str] = None
        self.drag_grab: Optional[Tuple[float, float]] = None
        self.drag_orig: Optional[Pose] = None
        self.status_msg = ""
        self.status_until = 0.0
        self.dirty = False
        self.edits: List[str] = []

    def update(self, dt: float) -> None:
        if self.playing:
            self.T += dt

    # ------------------------------------------------------------------ #
    # coordinate maps / geometry helpers
    def w2s(self, wx: float, wy: float) -> Tuple[int, int]:
        return int(self.canvas_cx + wx * self.ppm), int(self.canvas_cy - wy * self.ppm)

    def s2w(self, sx: float, sy: float) -> Tuple[float, float]:
        return (sx - self.canvas_cx) / self.ppm, (self.canvas_cy - sy) / self.ppm

    @property
    def ppm(self) -> float:
        return self.scenario.pixels_per_meter

    def subwin_rect(self) -> pygame.Rect:
        return pygame.Rect(0, self.H - self.SUBWIN_H, self.W, self.SUBWIN_H)

    def plot_rect(self) -> pygame.Rect:
        sw = self.subwin_rect()
        return pygame.Rect(sw.x + 55, sw.y + 56, 470, self.SUBWIN_H - 96)

    def field_specs(self, m):
        if m.type == "go_straight":
            return [("intercept", "speed"), ("duration", "duration")]
        if m.type in ("accelerate", "decelerate"):
            return [("intercept", "v0"), ("slope", "accel"), ("duration", "duration")]
        if m.type in ("turn_left", "turn_right"):
            return [("radius", "radius"), ("angle", "angle"), ("duration", "duration")]
        return [("duration", "duration")]

    def field_rects(self, m) -> dict:
        sw = self.subwin_rect()
        x = sw.x + 640
        return {name: pygame.Rect(x, sw.y + 56 + i * 42, 90, 26)
                for i, (name, _l) in enumerate(self.field_specs(m))}

    def field_label(self, m, name) -> str:
        return next((l for n, l in self.field_specs(m) if n == name), name)

    def header_buttons(self) -> dict:
        sw = self.subwin_rect()
        y = sw.y + 12
        return {"type": pygame.Rect(sw.right - 420, y, 160, 26),
                "prev": pygame.Rect(sw.right - 254, y, 48, 26),
                "next": pygame.Rect(sw.right - 200, y, 48, 26),
                "add_mvr": pygame.Rect(sw.right - 146, y, 64, 26),
                "del_mvr": pygame.Rect(sw.right - 76, y, 64, 26)}

    def cur_maneuver(self):
        if self.selected is None:
            return None
        a = self.scenario.actors[self.selected]
        return a.maneuvers[self.man_index] if a.maneuvers else None

    def set_status(self, msg: str) -> None:
        self.status_msg = msg
        self.status_until = self.T + 3.0

    def _mark(self, summary: str) -> None:
        self.dirty = True
        self.edits.append(summary)
        self.set_status(summary)

    def _resim(self) -> None:
        self.scenario.simulate()

    # ------------------------------------------------------------------ #
    # editing operations
    def next_actor_id(self) -> str:
        nums = [int(a.id) for a in self.scenario.actors if str(a.id).isdigit()]
        return str(max(nums) + 1) if nums else "0"

    def do_add_actor(self) -> None:
        i = len(self.scenario.actors)
        sx, sy, hd = ADD_PRESETS[i % len(ADD_PRESETS)]
        a = se.Actor(id=self.next_actor_id(), color=ADD_PALETTE[i % len(ADD_PALETTE)],
                     length=4.5, width=2.0, start=(sx, sy, hd),
                     maneuvers=[se.Maneuver(type="go_straight", duration=8.0,
                                            intercept=12.0, slope=0.0)])
        self.scenario.actors.append(a)
        self._resim()
        self.selected = len(self.scenario.actors) - 1
        self.man_index = 0
        self._mark(f"added actor {a.id}")

    def do_remove_actor(self) -> None:
        if self.selected is None:
            return
        a = self.scenario.actors.pop(self.selected)
        self._resim()
        self.selected = None
        self.man_index = 0
        self._mark(f"removed actor {a.id}")

    def do_add_maneuver(self) -> None:
        if self.selected is None:
            return
        a = self.scenario.actors[self.selected]
        v_in = a.maneuvers[self.man_index].exit_speed() if a.maneuvers else 10.0
        new_m = se.Maneuver(type="go_straight", duration=2.0,
                            intercept=(v_in if v_in > 0 else 10.0), slope=0.0)
        at = self.man_index + 1 if a.maneuvers else 0
        a.maneuvers.insert(at, new_m)
        self._resim()
        self.man_index = at
        self._mark(f"added maneuver to actor {a.id}")

    def do_del_maneuver(self) -> None:
        if self.selected is None:
            return
        a = self.scenario.actors[self.selected]
        if len(a.maneuvers) <= 1:
            self.set_status("cannot delete the last maneuver")
            return
        a.maneuvers.pop(self.man_index)
        self._resim()
        self.man_index = min(self.man_index, len(a.maneuvers) - 1)
        self._mark(f"deleted maneuver from actor {a.id}")

    def do_cycle_maneuver_type(self) -> None:
        if self.selected is None:
            return
        a = self.scenario.actors[self.selected]
        m = a.maneuvers[self.man_index]
        old = m.type
        idx = MANEUVER_TYPE_CYCLE.index(old) if old in MANEUVER_TYPE_CYCLE else -1
        m.type = MANEUVER_TYPE_CYCLE[(idx + 1) % len(MANEUVER_TYPE_CYCLE)]
        v_in = (a.maneuvers[self.man_index - 1].exit_speed()
                if self.man_index > 0 else 10.0)
        if m.type in ("turn_left", "turn_right"):
            if m.radius <= 0:
                m.radius = 5.0
            if m.angle <= 0:
                m.angle = 90.0
        if m.type in LONGITUDINAL_TYPES:
            m.intercept = v_in
            m.slope = (0.0 if m.type == "go_straight"
                       else 2.0 if m.type == "accelerate" else -2.0)
        if m.type == "stop":
            m.slope, m.intercept = 0.0, 0.0
        self._resim()
        self.focus_field = None
        self._mark(f"actor {a.id} maneuver {self.man_index}: {old} -> {m.type}")

    def commit_spawn_edit(self) -> None:
        if self.selected is None or self.drag_orig is None:
            return
        a = self.scenario.actors[self.selected]
        if a.start != self.drag_orig:
            self._mark(f"moved actor {a.id} spawn")

    def apply_param(self, param: str, new_val: float) -> None:
        a = self.scenario.actors[self.selected]
        m = a.maneuvers[self.man_index]
        old = getattr(m, param)
        if param == "duration":
            new_val = max(0.05, new_val)
        elif param == "radius":
            new_val = max(0.5, new_val)
        elif param == "angle":
            new_val = min(179.0, max(1.0, new_val))
        elif param == "intercept":
            new_val = max(0.0, new_val)
        setattr(m, param, new_val)
        self._resim()
        self._mark(f"{a.id}.{m.type}.{param}: {old:.3g} -> {new_val:.3g}")

    def plot_maps(self, m, pr: pygame.Rect):
        tmax = max(1e-6, m.duration)
        v0, v1 = m.intercept, m.intercept + m.slope * m.duration
        vmax = max(v0, v1, 1.0) * 1.15
        vmin = min(v0, v1, 0.0) - 0.15 * abs(max(v0, v1, 1.0))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        def t2x(t): return pr.x + (t / tmax) * pr.width
        def v2y(v): return pr.bottom - (v - vmin) / (vmax - vmin) * pr.height
        def y2v(y): return vmin + (pr.bottom - y) / pr.height * (vmax - vmin)
        return t2x, v2y, y2v, tmax, vmin, vmax

    # ------------------------------------------------------------------ #
    # drawing
    def draw(self, surf: pygame.Surface) -> None:
        self._s = surf
        self._draw_map()
        for i, a in enumerate(self.scenario.actors):
            self._draw_actor(i, a)
        self._draw_subwindow()
        self._draw_topbar()

    def _draw_map(self) -> None:
        s = self._s
        s.fill(C_GRASS)
        arm = self.scenario.map.arm_length
        half = self.scenario.map.lane_width
        for (a, b) in (((-arm, half), (arm, -half)), ((-half, arm), (half, -arm))):
            x0, y0 = self.w2s(*a)
            x1, y1 = self.w2s(*b)
            pygame.draw.rect(s, C_ROAD, pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        dash, d = 3.0, -arm
        while d < arm:
            if abs(d) > half:
                pygame.draw.line(s, C_LINE, self.w2s(d, 0),
                                 self.w2s(min(d + dash, arm), 0), 2)
                pygame.draw.line(s, C_LINE, self.w2s(0, d),
                                 self.w2s(0, min(d + dash, arm)), 2)
            d += dash * 2
        for sx, sy, ex, ey in [(-half, -half, 0, -half), (0, half, half, half),
                               (-half, half, -half, 0), (half, -half, half, 0)]:
            pygame.draw.line(s, C_EDGE, self.w2s(sx, sy), self.w2s(ex, ey), 2)

    def _actor_corners_world(self, pose: Pose, a):
        x, y, hd = pose
        h = math.radians(hd)
        fx, fy = math.cos(h), math.sin(h)
        px, py = -math.sin(h), math.cos(h)
        L, W = a.length / 2, a.width / 2
        return [(x + fx * L + px * W, y + fy * L + py * W),
                (x + fx * L - px * W, y + fy * L - py * W),
                (x - fx * L - px * W, y - fy * L - py * W),
                (x - fx * L + px * W, y - fy * L + py * W)]

    def _rotation_handle_world(self, a):
        sx, sy, hd = a.start
        h = math.radians(hd)
        r = a.length / 2 + 2.5
        return (sx + math.cos(h) * r, sy + math.sin(h) * r)

    def _point_in_pose(self, a, pose: Pose, wx: float, wy: float) -> bool:
        cx, cy, hd = pose
        h = math.radians(hd)
        dx, dy = wx - cx, wy - cy
        along = dx * math.cos(h) + dy * math.sin(h)
        lat = -dx * math.sin(h) + dy * math.cos(h)
        return abs(along) <= a.length / 2 + 0.5 and abs(lat) <= a.width / 2 + 0.5

    def _draw_actor(self, idx: int, a) -> None:
        s = self._s
        phase = self.T % self.scenario.period
        x, y, hd = a.pose_at_time(phase)
        pts = [self.w2s(*c) for c in self._actor_corners_world((x, y, hd), a)]
        pygame.draw.polygon(s, tuple(a.color), pts)
        pygame.draw.polygon(s, (20, 20, 20), pts, 1)
        pygame.draw.line(s, (250, 250, 250), pts[0], pts[1], 3)
        if idx == self.selected:
            pygame.draw.polygon(s, C_SEL, pts, 3)
            if not self.playing:
                spts = [self.w2s(*c) for c in self._actor_corners_world(a.start, a)]
                pygame.draw.polygon(s, C_SEL, spts, 2)
                sc0 = self.w2s(a.start[0], a.start[1])
                hpt = self.w2s(*self._rotation_handle_world(a))
                pygame.draw.line(s, C_SEL, sc0, hpt, 2)
                pygame.draw.circle(s, C_SEL, hpt, 6)
                tag = self.font_sm.render("spawn (drag to move, handle to rotate)",
                                          True, C_SEL)
                s.blit(tag, (spts[3][0], spts[3][1] + 4))
        label = self.font_sm.render(str(a.id), True, C_TEXT)
        lp = self.w2s(x, y)
        s.blit(label, (lp[0] - label.get_width() // 2, lp[1] - 8))

    def _draw_button(self, rect, label, active=False, enabled=True) -> None:
        s = self._s
        col = C_BTN_HL if active else C_BTN
        if not enabled:
            col = (44, 46, 52)
        pygame.draw.rect(s, col, rect, border_radius=6)
        pygame.draw.rect(s, (90, 94, 105), rect, 1, border_radius=6)
        txt = self.font.render(label, True, C_TEXT if enabled else (120, 120, 130))
        s.blit(txt, (rect.centerx - txt.get_width() // 2,
                     rect.centery - txt.get_height() // 2))

    def _draw_topbar(self) -> None:
        s = self._s
        pygame.draw.rect(s, C_BAR, pygame.Rect(0, 0, self.W, self.TOPBAR_H))
        self._draw_button(self.btn_reset, "Reset")
        self._draw_button(self.btn_play, "Pause" if self.playing else "Play",
                          active=self.playing)
        self._draw_button(self.btn_add, "+ Actor")
        self._draw_button(self.btn_del, "- Actor", enabled=(self.selected is not None))
        s.blit(self.font.render("T=", True, C_TEXT), (18, 18))
        focused = (self.focus_field == "time")
        pygame.draw.rect(s, (18, 20, 26), self.time_field)
        pygame.draw.rect(s, C_BTN_HL if focused else (90, 94, 105), self.time_field, 2)
        shown = self.edit_buffer if focused else f"{self.T % self.scenario.period:.2f}"
        s.blit(self.font.render(shown, True, C_TEXT),
               (self.time_field.x + 6, self.time_field.y + 5))
        info = f"/ {self.scenario.period:.2f}s   {'PLAYING' if self.playing else 'PAUSED'}"
        s.blit(self.font.render(info, True, C_TEXT), (self.time_field.right + 10, 18))

    def _draw_subwindow(self) -> None:
        s = self._s
        sw = self.subwin_rect()
        pygame.draw.rect(s, C_PANEL, sw)
        pygame.draw.line(s, (80, 84, 95), (sw.x, sw.y), (sw.right, sw.y), 2)
        if self.status_msg and self.T < self.status_until:
            st = self.font_sm.render(self.status_msg, True, (150, 220, 150))
            s.blit(st, (sw.right - st.get_width() - 16, sw.bottom - 26))
        m = self.cur_maneuver()
        if self.playing or self.selected is None or m is None:
            hint = "Pause and click an actor to edit its timing curve; " \
                   "+Actor / -Actor add or remove."
            s.blit(self.font.render(hint, True, (150, 154, 165)), (sw.x + 20, sw.y + 22))
            return
        a = self.scenario.actors[self.selected]
        title = f"Actor {a.id}   maneuver {self.man_index + 1}/{len(a.maneuvers)}"
        s.blit(self.font_big.render(title, True, C_TEXT), (sw.x + 16, sw.y + 14))
        hb = self.header_buttons()
        self._draw_button(hb["type"], f"type: {m.type}")
        self._draw_button(hb["prev"], "prev")
        self._draw_button(hb["next"], "next")
        self._draw_button(hb["add_mvr"], "+mvr")
        self._draw_button(hb["del_mvr"], "-mvr", enabled=(len(a.maneuvers) > 1))
        pr = self.plot_rect()
        if m.curve_kind == "velocity":
            pygame.draw.rect(s, (18, 20, 26), pr)
            pygame.draw.rect(s, C_AXIS, pr, 1)
            t2x, v2y, _, tmax, vmin, vmax = self.plot_maps(m, pr)
            s.blit(self.font_sm.render("velocity (m/s)", True, C_AXIS), (pr.x - 4, pr.y - 18))
            s.blit(self.font_sm.render("time (s)", True, C_AXIS),
                   (pr.right - 60, pr.bottom + 6))
            if vmin < 0 < vmax:
                zy = v2y(0)
                pygame.draw.line(s, (70, 74, 85), (pr.x, zy), (pr.right, zy), 1)
            p_left = (t2x(0), v2y(m.intercept))
            p_right = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
            pygame.draw.line(s, C_CURVE, p_left, p_right, 2)
            pygame.draw.circle(s, C_SEL, (int(p_left[0]), int(p_left[1])), 6)
            if m.type != "go_straight":
                pygame.draw.circle(s, C_SEL, (int(p_right[0]), int(p_right[1])), 6)
            phase = self.T % self.scenario.period
            if a.cum[self.man_index] <= phase < a.cum[self.man_index + 1]:
                mx = t2x(phase - a.cum[self.man_index])
                pygame.draw.line(s, (250, 120, 120), (mx, pr.y), (mx, pr.bottom), 1)
        else:
            note = {"turn_left": "left turn - set radius, angle, duration",
                    "turn_right": "right turn - set radius, angle, duration",
                    "stop": "stop - hold position for duration"}.get(m.type, "")
            s.blit(self.font.render(note, True, (150, 154, 165)), (pr.x, pr.y + 8))
        for name, rect in self.field_rects(m).items():
            s.blit(self.font.render(self.field_label(m, name), True, C_TEXT),
                   (rect.x - 90, rect.y + 4))
            focused = (self.focus_field == name)
            pygame.draw.rect(s, (18, 20, 26), rect)
            pygame.draw.rect(s, C_BTN_HL if focused else (90, 94, 105), rect, 2)
            shown = self.edit_buffer if focused else f"{getattr(m, name):.4g}"
            s.blit(self.font.render(shown, True, C_TEXT), (rect.x + 6, rect.y + 4))

    # ------------------------------------------------------------------ #
    # events (mouse coords are pane-local)
    def handle_event(self, event, local_pos: Optional[Tuple[int, int]] = None) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(*local_pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging in ("spawn", "rotate"):
                self.commit_spawn_edit()
            self.dragging = None
            self.drag_grab = None
            self.drag_orig = None
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._handle_drag(*local_pos)
        elif event.type == pygame.KEYDOWN:
            self._handle_key(event)

    def _handle_key(self, event) -> None:
        if self.focus_field is not None:
            if event.key == pygame.K_RETURN:
                self._commit_field()
            elif event.key == pygame.K_ESCAPE:
                self.focus_field = None
                self.edit_buffer = ""
            elif event.key == pygame.K_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif event.unicode in "0123456789.-+eE":
                self.edit_buffer += event.unicode
        else:
            if event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.key == pygame.K_ESCAPE:
                self.selected = None
            elif event.key in (pygame.K_DELETE,) and self.selected is not None:
                self.do_remove_actor()
            elif event.key == pygame.K_a:
                self.do_add_actor()

    def _commit_field(self) -> None:
        if self.focus_field is None:
            return
        try:
            val = float(self.edit_buffer)
            if self.focus_field == "time":
                self.T = max(0.0, val)
            else:
                self.apply_param(self.focus_field, val)
        except ValueError:
            self.set_status("invalid number")
        self.focus_field = None
        self.edit_buffer = ""

    def _handle_click(self, mx, my) -> None:
        if self.btn_play.collidepoint(mx, my):
            self.playing = not self.playing
            return
        if self.btn_reset.collidepoint(mx, my):
            self.T = 0.0
            return
        if self.time_field.collidepoint(mx, my):
            self.playing = False
            self.focus_field = "time"
            self.edit_buffer = ""
            return
        if self.btn_add.collidepoint(mx, my):
            self.do_add_actor()
            return
        if self.btn_del.collidepoint(mx, my) and self.selected is not None:
            self.do_remove_actor()
            return
        if not self.playing and self.selected is not None and self.cur_maneuver():
            sw = self.subwin_rect()
            hb = self.header_buttons()
            a = self.scenario.actors[self.selected]
            if hb["prev"].collidepoint(mx, my):
                self.man_index = (self.man_index - 1) % len(a.maneuvers)
                self.focus_field = None
                return
            if hb["next"].collidepoint(mx, my):
                self.man_index = (self.man_index + 1) % len(a.maneuvers)
                self.focus_field = None
                return
            if hb["add_mvr"].collidepoint(mx, my):
                self.do_add_maneuver()
                self.focus_field = None
                return
            if hb["del_mvr"].collidepoint(mx, my):
                self.do_del_maneuver()
                self.focus_field = None
                return
            if hb["type"].collidepoint(mx, my):
                self.do_cycle_maneuver_type()
                return
            m = self.cur_maneuver()
            for key, rect in self.field_rects(m).items():
                if rect.collidepoint(mx, my):
                    self.focus_field = key
                    self.edit_buffer = ""
                    return
            if m.curve_kind == "velocity":
                pr = self.plot_rect()
                t2x, v2y, _, tmax, _, _ = self.plot_maps(m, pr)
                pl = (t2x(0), v2y(m.intercept))
                prg = (t2x(tmax), v2y(m.intercept + m.slope * tmax))
                if (mx - pl[0]) ** 2 + (my - pl[1]) ** 2 < 100:
                    self.dragging = "left"
                    self.focus_field = None
                    return
                if m.type != "go_straight" and (mx - prg[0]) ** 2 + (my - prg[1]) ** 2 < 100:
                    self.dragging = "right"
                    self.focus_field = None
                    return
            if sw.collidepoint(mx, my):
                return
        if not self.playing and self.TOPBAR_H < my < self.TOPBAR_H + self.CANVAS_H:
            wx, wy = self.s2w(mx, my)
            if self.selected is not None:
                a = self.scenario.actors[self.selected]
                hpt = self.w2s(*self._rotation_handle_world(a))
                if (mx - hpt[0]) ** 2 + (my - hpt[1]) ** 2 < 100:
                    self.dragging = "rotate"
                    self.drag_orig = a.start
                    self.focus_field = None
                    self.T = 0.0
                    return
                if self._point_in_pose(a, a.start, wx, wy):
                    self.dragging = "spawn"
                    self.drag_grab = (wx, wy)
                    self.drag_orig = a.start
                    self.focus_field = None
                    self.T = 0.0
                    return
            phase = self.T % self.scenario.period
            for i, a in enumerate(self.scenario.actors):
                if self._point_in_pose(a, a.pose_at_time(phase), wx, wy):
                    self.selected = i
                    self.man_index = a.active_index(phase) if a.total > phase else 0
                    self.focus_field = None
                    return

    def _handle_drag(self, mx, my) -> None:
        if self.dragging in ("spawn", "rotate") and self.selected is not None:
            a = self.scenario.actors[self.selected]
            wx, wy = self.s2w(mx, my)
            if self.dragging == "spawn" and self.drag_grab is not None:
                nx = self.drag_orig[0] + (wx - self.drag_grab[0])
                ny = self.drag_orig[1] + (wy - self.drag_grab[1])
                a.start = (nx, ny, a.start[2])
            else:
                hd = math.degrees(math.atan2(wy - a.start[1], wx - a.start[0]))
                a.start = (a.start[0], a.start[1], round(hd, 1))
            self._resim()
            self.dirty = True
            return
        m = self.cur_maneuver()
        if m is None or m.curve_kind != "velocity":
            return
        pr = self.plot_rect()
        _, _, y2v, tmax, _, _ = self.plot_maps(m, pr)
        val = y2v(min(max(my, pr.y), pr.bottom))
        if self.dragging == "left":
            self.apply_param("intercept", val)
        elif self.dragging == "right":
            self.apply_param("slope", (val - m.intercept) / max(1e-6, m.duration))
