#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/session_editor.py — the session editor: the v0-style realization editor as a
pane, plus a side pane that drives the closed-loop game tree.

Left pane (session): live D1/D2/D3 verdict of the current realization, the game
tree (nodes colored by kind, click to navigate), and controls —
  Move forward   advance the rollout to the next decision point (orchestrator
                 intervenes, or a collision/infeasible terminal) → new child node
  Commit pert    commit the edits you made in the editor as a perturbation node
  Discard        drop uncommitted edits (reload the current node)
  Checkpoint     save the current realization as a checkpoint node
  Propose        ask the orchestrator for an intervention (a proposal branch)

Right pane (realization editor, unchanged behavior): pause and edit the current
node's realization — select/drag actors, edit maneuvers, velocity curves,
+Actor/-Actor. Edits are explicit-commit (choice 3b): they don't enter the tree
until you press Commit pert (or Checkpoint).

Usage:  python3 session_editor.py [--base scenario_v20.yaml] [--session ID]
Requires pygame.  Build the HTML tree afterward with build_tree.py.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import orchestrator as O              # noqa: E402
import directives_script as ds        # noqa: E402
import scenario_editor as se          # noqa: E402
from realization_editor import RealizationEditor   # noqa: E402

TREE_W = 340
EDITOR_W = 1000
EDITOR_H = 820
W, H = TREE_W + EDITOR_W, EDITOR_H

BG = (16, 18, 24)
PANEL = (26, 29, 37)
TXT = (226, 232, 240)
MUTED = (140, 150, 166)
BTN = (54, 58, 70)
BTN_HL = (80, 120, 200)
BTN_OFF = (40, 42, 48)
KIND_COLOR = {"start": (138, 148, 166), "intervention": (56, 178, 198),
              "perturbation": (230, 162, 60), "checkpoint": (212, 175, 55),
              "proposal": (160, 108, 213), "infeasible": (224, 82, 82),
              "collision": (88, 194, 106)}


def _copy_sc(sc: se.Scenario) -> se.Scenario:
    new = copy.deepcopy(sc)
    new.simulate()
    return new


class SessionEditor:
    def __init__(self, base: str, session_dir: str):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("v4 session editor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.font_sm = pygame.font.SysFont("consolas,menlo,monospace", 12)
        self.font_big = pygame.font.SysFont("consolas,menlo,monospace", 17, bold=True)

        self.orch = O.Orchestrator(base, session_dir, ego="0",
                                   label="v20 red-light (session)")
        self.editor = RealizationEditor(_copy_sc(self.orch.sc), EDITOR_W, EDITOR_H)
        self.banner = "Edit the realization, then Commit pert / Checkpoint / Move forward."
        self.scroll = 0
        self.frame = 0
        self.verdict = None
        self._node_rects = {}

        y = 108
        self.buttons = {}
        for key, label in [("forward", "Move forward  →"), ("commit", "Commit pert"),
                           ("discard", "Discard edits"), ("checkpoint", "Checkpoint"),
                           ("propose", "Propose")]:
            self.buttons[key] = pygame.Rect(12, y, TREE_W - 24, 30)
            y += 36
        self.tree_top = y + 6
        self._recompute_verdict()

    # ---- current-realization verdict (reflects live edits) ---- #
    def _recompute_verdict(self):
        self.verdict = ds.evaluate(_copy_sc(self.editor.scenario),
                                   self.orch.ego, self.orch.signals, self.orch.prm)

    # ---- session actions ---- #
    def _sync_editor_to_current(self):
        self.editor.set_scenario(_copy_sc(self.orch.sc))
        self._recompute_verdict()

    def act_commit(self):
        if not self.editor.dirty:
            self.banner = "No edits to commit."
            return
        summary = "; ".join(self.editor.edits[-3:]) or "manual edit"
        v = self.orch.commit_perturbation(_copy_sc(self.editor.scenario), summary)
        self._sync_editor_to_current()
        self.banner = f"committed perturbation v{v}"

    def act_discard(self):
        self._sync_editor_to_current()
        self.banner = "edits discarded (reloaded current node)"

    def act_checkpoint(self):
        v = self.orch.checkpoint(scenario=_copy_sc(self.editor.scenario))
        self._sync_editor_to_current()
        self.banner = f"checkpoint saved v{v}"

    def act_forward(self):
        if self.editor.dirty:
            self.banner = "Uncommitted edits — Commit pert or Discard first."
            return
        v = self.orch.advance_to_decision()
        self._sync_editor_to_current()
        if v is None:
            self.banner = "no decision reached (terminal or budget)"
        else:
            rec = self.orch.store.versions[-1]
            self.banner = f"advanced to v{rec['version']} ({rec['kind']}) — {rec['delta'][:40]}"

    def act_propose(self):
        if self.editor.dirty:
            self.banner = "Commit or discard edits before proposing."
            return
        rr = self.orch.compute_repair()
        if not (rr.feasible and rr.interventions):
            self.banner = "orchestrator proposes nothing (family holds or infeasible)"
            return
        trial = self.orch.build_trial(rr)
        ver = self.orch.save_proposal(rr, trial)
        self.orch.load_checkpoint(ver)           # navigate to the proposal to view it
        self._sync_editor_to_current()
        self.banner = f"proposal v{ver}: " + "; ".join(str(i) for i in rr.interventions)[:46]

    def load_node(self, version: int):
        self.orch.load_checkpoint(version)
        self._sync_editor_to_current()
        rec = next(v for v in self.orch.store.versions if v["version"] == version)
        self.banner = f"loaded v{version} ({rec['kind']}) @ {rec['sim_time']}s"

    # ---- tree layout ---- #
    def _layout_tree(self):
        versions = self.orch.store.versions
        V = {v["version"]: dict(v, kids=[]) for v in versions}
        roots = []
        for n in V.values():
            (V[n["parent"]]["kids"].append(n) if n["parent"] in V else roots.append(n))
        for n in V.values():
            n["kids"].sort(key=lambda c: c["version"])
        NW, NH, HG, VG = 50, 26, 12, 22
        cur = [0]

        def place(n, d):
            n["d"] = d
            if not n["kids"]:
                n["tx"] = cur[0]
                cur[0] += NW + HG
            else:
                for c in n["kids"]:
                    place(c, d + 1)
                n["tx"] = (n["kids"][0]["tx"] + n["kids"][-1]["tx"]) / 2
        for r in sorted(roots, key=lambda c: c["version"]):
            place(r, d=0)
        return V, NW, NH, HG, VG

    # ---- drawing ---- #
    def _draw_button(self, rect, label, enabled=True, hot=False):
        col = BTN_HL if hot else (BTN if enabled else BTN_OFF)
        pygame.draw.rect(self.screen, col, rect, border_radius=6)
        pygame.draw.rect(self.screen, (90, 94, 105), rect, 1, border_radius=6)
        c = TXT if enabled else (120, 122, 130)
        t = self.font.render(label, True, c)
        self.screen.blit(t, (rect.centerx - t.get_width() // 2,
                             rect.centery - t.get_height() // 2))

    def _draw_tree_pane(self):
        s = self.screen
        pygame.draw.rect(s, PANEL, pygame.Rect(0, 0, TREE_W, H))
        pygame.draw.line(s, (60, 64, 74), (TREE_W, 0), (TREE_W, H), 2)
        s.blit(self.font_big.render("Session — game tree", True, TXT), (12, 8))
        # verdict lamps
        v = self.verdict
        if v is not None:
            for i, (nm, ev) in enumerate((("D1", v.d1), ("D2", v.d2), ("D3", v.d3))):
                c = (88, 194, 106) if ev.value else (224, 82, 82)
                r = pygame.Rect(12 + i * 60, 34, 54, 22)
                pygame.draw.rect(s, c, r, border_radius=4)
                s.blit(self.font_sm.render(f"{nm}:{'OK' if ev.value else 'X'}", True,
                                           (10, 10, 10)), (r.x + 6, r.y + 4))
            info = f"hero {v.hero}  t*={v.t_star:.1f}" if v.t_star else f"hero {v.hero}"
            s.blit(self.font_sm.render(info, True, MUTED), (200, 38))
        cur_rec = next((x for x in self.orch.store.versions
                        if x["version"] == self.orch.parent), None)
        if cur_rec:
            s.blit(self.font_sm.render(
                f"current: v{cur_rec['version']} {cur_rec['kind']} "
                f"@ {cur_rec['sim_time']}s", True, MUTED), (12, 62))
        if self.editor.dirty:
            s.blit(self.font_sm.render("● uncommitted edits", True, (230, 162, 60)),
                   (12, 80))
        # buttons
        dirty = self.editor.dirty
        self._draw_button(self.buttons["forward"], "Move forward  →", enabled=not dirty)
        self._draw_button(self.buttons["commit"], "Commit pert", enabled=dirty)
        self._draw_button(self.buttons["discard"], "Discard edits", enabled=dirty)
        self._draw_button(self.buttons["checkpoint"], "Checkpoint")
        self._draw_button(self.buttons["propose"], "Propose", enabled=not dirty)
        # tree
        self._draw_tree()
        # banner
        pygame.draw.rect(s, (20, 22, 28), pygame.Rect(0, H - 34, TREE_W, 34))
        for i, line in enumerate(self._wrap(self.banner, 42)[:2]):
            s.blit(self.font_sm.render(line, True, MUTED), (10, H - 30 + i * 14))

    def _wrap(self, text, n):
        out, cur = [], ""
        for w in text.split():
            if len(cur) + len(w) + 1 > n:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            out.append(cur)
        return out

    def _draw_tree(self):
        s = self.screen
        area = pygame.Rect(6, self.tree_top, TREE_W - 12, H - 34 - self.tree_top)
        prev_clip = s.get_clip()
        s.set_clip(area)
        V, NW, NH, HG, VG = self._layout_tree()
        self._node_rects = {}
        if V:
            maxx = max(n["tx"] for n in V.values()) + NW
            sx = min(1.0, (area.width - 8) / maxx) if maxx > 0 else 1.0
            for n in V.values():
                x = area.x + 4 + n["tx"] * sx
                y = area.y + 6 + n["d"] * (NH + VG) - self.scroll
                w = max(22, NW * sx)
                self._node_rects[n["version"]] = pygame.Rect(int(x), int(y), int(w), NH)
            # edges
            for n in V.values():
                for c in n["kids"]:
                    r1, r2 = self._node_rects[n["version"]], self._node_rects[c["version"]]
                    col = KIND_COLOR.get(c["kind"], (90, 94, 105))
                    pygame.draw.line(s, col, (r1.centerx, r1.bottom),
                                     (r2.centerx, r2.top), 2)
            # nodes
            for n in V.values():
                r = self._node_rects[n["version"]]
                col = KIND_COLOR.get(n["kind"], (90, 94, 105))
                pygame.draw.rect(s, (34, 38, 48), r, border_radius=4)
                width = 3 if n["version"] == self.orch.parent else 1
                bcol = (255, 255, 255) if n["version"] == self.orch.parent else col
                pygame.draw.rect(s, bcol, r, width, border_radius=4)
                lab = self.font_sm.render(f"v{n['version']}", True, col)
                s.blit(lab, (r.x + 4, r.y + 4))
        s.set_clip(prev_clip)

    def render(self):
        self.screen.fill(BG)
        pane = self.screen.subsurface(pygame.Rect(TREE_W, 0, EDITOR_W, EDITOR_H))
        self.editor.draw(pane)
        self._draw_tree_pane()
        pygame.display.flip()

    # ---- events ---- #
    def _tree_click(self, mx, my):
        for key, r in self.buttons.items():
            if r.collidepoint(mx, my):
                {"forward": self.act_forward, "commit": self.act_commit,
                 "discard": self.act_discard, "checkpoint": self.act_checkpoint,
                 "propose": self.act_propose}[key]()
                self._recompute_verdict()
                return
        for ver, r in self._node_rects.items():
            if r.collidepoint(mx, my):
                self.load_node(ver)
                return

    def loop(self):
        alive = True
        self._recompute_verdict()
        while alive:
            dt = self.clock.tick(60) / 1000.0
            self.editor.update(dt)
            self.frame += 1
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if mx < TREE_W:
                        self.scroll = max(0, self.scroll - e.y * 30)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    mx, my = e.pos
                    if mx < TREE_W:
                        self._tree_click(mx, my)
                    else:
                        self.editor.handle_event(e, (mx - TREE_W, my))
                        self._recompute_verdict()
                elif e.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                    mx, my = e.pos
                    self.editor.handle_event(e, (mx - TREE_W, my))
                    if e.type == pygame.MOUSEBUTTONUP:
                        self._recompute_verdict()
                elif e.type == pygame.KEYDOWN:
                    self.editor.handle_event(e)
                    self._recompute_verdict()
            # keep the verdict fresh during curve drags
            if self.editor.dragging and self.frame % 4 == 0:
                self._recompute_verdict()
            self.render()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(O.V2, "scenarios",
                                                   "scenario_v20.yaml"))
    ap.add_argument("--session", default=None)
    args = ap.parse_args()
    sid = args.session or (datetime.now().strftime("%Y%m%dT%H%M%S") + "_session")
    session_dir = os.path.join(HERE, "sessions", sid)
    SessionEditor(args.base, session_dir).loop()
    print(f"session saved: {session_dir}\n"
          f"build the tree:  python3 build_tree.py sessions/{sid}")


if __name__ == "__main__":
    main()
