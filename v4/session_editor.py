#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4/session_editor.py — interactive pygame front end for closed-loop orchestration.

Drives the v4 kernel (orchestrator.Orchestrator): run/freeze the loop, step,
scrub the forward realization while frozen, inject perturbations (including on the
ego), save/load checkpoints, and ask the orchestrator to propose an intervention.
Every decision point is written to the session folder; run `build_tree.py` on it
afterward for the colored game tree.

Keys
  Space     run / freeze (freeze time)
  .         single step (while frozen)
  ← →       scrub forward realization (while frozen)
  click     select an actor
  a         add actor  (cycles spawn leg SE→EN→WS→NW; straight-through)
  t         set selected actor's route  (cycles straight→left→right)  [may be the ego]
  s / S     slow selected actor to 3 m/s  /  speed it to 12 m/s
  Del       remove selected actor
  k         save checkpoint
  l         load next checkpoint (branch from it)
  p         propose intervention (ghost + proposal node); Enter accept, Esc discard
  q / Esc   quit  (Esc discards a pending proposal first)

Usage:  python3 session_editor.py [--base scenario_v20.yaml] [--session ID]
Requires pygame.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import orchestrator as O          # noqa: E402
import scenario_editor as se      # noqa: E402
import directives as dv           # noqa: E402

# layout
BEV = 720
VIEW = 42.0
SCALE = BEV / (2 * VIEW)
TOPBAR, BOTBAR = 72, 46
W, H = BEV, TOPBAR + BEV + BOTBAR

GRASS = (106, 153, 78)
ASPHALT = (70, 70, 74)
LINE = (232, 232, 232)
BG = (24, 24, 28)
PANEL = (30, 34, 44)
TXT = (226, 232, 240)
MUTED = (138, 148, 166)
KIND_COLOR = {"start": (138, 148, 166), "intervention": (56, 178, 198),
              "perturbation": (230, 162, 60), "checkpoint": (212, 175, 55),
              "proposal": (160, 108, 213), "infeasible": (224, 82, 82),
              "collision": (88, 194, 106)}
LEGS = ["SE", "EN", "WS", "NW"]
TURNS = ["straight", "left", "right"]


def w2p(x, y):
    return (int(BEV / 2 + x * SCALE), int(TOPBAR + BEV / 2 - y * SCALE))


def rot_rect(x, y, hdg, length, width):
    import math
    h = math.radians(hdg)
    dx, dy = math.cos(h), math.sin(h)
    px, py = -dy, dx
    hl, hw = length / 2, width / 2
    return [w2p(x + dx * hl + px * hw, y + dy * hl + py * hw),
            w2p(x + dx * hl - px * hw, y + dy * hl - py * hw),
            w2p(x - dx * hl - px * hw, y - dy * hl - py * hw),
            w2p(x - dx * hl + px * hw, y - dy * hl + py * hw)]


class Editor:
    def __init__(self, base, session_dir):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("v4 — closed-loop scenario orchestration")
        self.f_big = pygame.font.SysFont("Menlo,Consolas,monospace", 18, bold=True)
        self.f = pygame.font.SysFont("Menlo,Consolas,monospace", 14)
        self.f_sm = pygame.font.SysFont("Menlo,Consolas,monospace", 12)
        self.orch = O.Orchestrator(base, session_dir, ego="0",
                                   label="v20 red-light (interactive)")
        self.running_loop = False
        self.acc = 0.0
        self.scrub = 0.0
        self.selected = None
        self.leg_i = 0
        self.proposal = None                 # (version, trial_sc, rr)
        self.msg = "Space=run  a/t/s=perturb  p=propose  k=checkpoint  l=load"
        self.clock = pygame.time.Clock()

    # ---- rendering ---- #
    def _tau(self):
        return max(0.0, self.orch.T - self.orch.t_base + self.scrub)

    def _live_sc(self):
        if self.proposal:
            return self.proposal[1]
        return self.orch.sc

    def draw_map(self, surf):
        lw = self.orch.sc.map.lane_width
        b = lw * SCALE
        cx, cy = w2p(0, 0)
        pygame.draw.rect(surf, GRASS, (0, TOPBAR, BEV, BEV))
        pygame.draw.rect(surf, ASPHALT, (cx - b, TOPBAR, 2 * b, BEV))
        pygame.draw.rect(surf, ASPHALT, (0, cy - b, BEV, 2 * b))
        for p in range(int(b + 1.6 * SCALE), int(BEV / 2), int(3.6 * SCALE)):
            for s in (1, -1):
                pygame.draw.line(surf, LINE, (cx, cy + s * p),
                                 (cx, cy + s * min(p + 2 * SCALE, BEV / 2)), 2)
                pygame.draw.line(surf, LINE, (cx + s * p, cy),
                                 (cx + s * min(p + 2 * SCALE, BEV / 2), cy), 2)
        sig = {"N": (0, lw), "S": (0, -lw), "E": (lw, 0), "W": (-lw, 0)}
        for arm, (sx, sy) in sig.items():
            px, py = w2p(sx, sy)
            vx, vy = px - cx, py - cy
            n = max((vx * vx + vy * vy) ** 0.5, 1e-6)
            px, py = px + vx / n * 16, py + vy / n * 16
            red = self.orch.signals.get(arm) == "red"
            pygame.draw.circle(surf, (220, 40, 40) if red else (60, 200, 80),
                               (int(px), int(py)), 6)

    def draw_actors(self, surf, ghost=False):
        sc = self._live_sc()
        sc.simulate()
        tau = self._tau()
        k = int(round(tau / se.DT))
        for a in sc.actors:
            kk = max(0, min(k, len(a.traj) - 1))
            x, y, hdg = a.traj[kk]
            col = tuple(a.color)
            if a.id == self.orch.ego:
                col = (90, 190, 110)
            pts = rot_rect(x, y, hdg, a.length, a.width)
            if ghost:
                s2 = surf.convert_alpha()
            pygame.draw.polygon(surf, col, pts)
            pygame.draw.polygon(surf, (15, 15, 15), pts, 1)
            if a.id == self.selected:
                pygame.draw.polygon(surf, (255, 255, 255), pts, 3)
            cx, cy = w2p(x, y)
            lab = self.f_sm.render(a.id, True, (0, 0, 0))
            surf.blit(lab, (cx - lab.get_width() // 2, cy - 7))

    def draw_topbar(self, surf):
        pygame.draw.rect(surf, BG, (0, 0, W, TOPBAR))
        res = self.orch.current()
        state = "RUN " if self.running_loop else "FROZEN"
        t = self.f_big.render(f"t={self.orch.T:5.2f}s  [{state}]", True, TXT)
        surf.blit(t, (10, 8))
        # verdict lamps
        x0 = 250
        for i, (name, ev) in enumerate((("D1", res.d1), ("D2", res.d2),
                                        ("D3", res.d3))):
            c = (88, 194, 106) if ev.value else (224, 82, 82)
            pygame.draw.rect(surf, c, (x0 + i * 78, 10, 70, 22), border_radius=5)
            surf.blit(self.f.render(f"{name}:{'OK' if ev.value else 'X'}", True,
                                    (12, 12, 12)), (x0 + i * 78 + 8, 13))
        info = f"hero={res.hero}  t*={res.t_star:.1f}" if res.t_star else "hero=-"
        surf.blit(self.f.render(info, True, MUTED), (x0 + 3 * 78 + 12, 13))
        surf.blit(self.f_sm.render(self.msg, True, MUTED), (10, 42))
        if self.proposal:
            surf.blit(self.f.render("PROPOSAL — Enter=accept  Esc=discard", True,
                                    KIND_COLOR["proposal"]), (W - 340, 42))

    def draw_botbar(self, surf):
        y = TOPBAR + BEV
        pygame.draw.rect(surf, PANEL, (0, y, W, BOTBAR))
        cps = [v for v in self.orch.store.versions
               if v["kind"] in ("start", "checkpoint")]
        surf.blit(self.f_sm.render("checkpoints:", True, MUTED), (8, y + 6))
        x = 108
        for v in cps:
            c = KIND_COLOR.get(v["kind"], MUTED)
            label = f"v{v['version']}·{v['sim_time']:.1f}s"
            r = self.f_sm.render(label, True, c)
            pygame.draw.rect(surf, (20, 24, 32),
                             (x - 4, y + 4, r.get_width() + 8, 18), border_radius=4)
            surf.blit(r, (x, y + 6))
            x += r.get_width() + 18
        n = len(self.orch.store.versions)
        surf.blit(self.f_sm.render(f"{n} decision nodes  ·  add=a route=t speed=s/S "
                                   f"del=Del", True, MUTED), (8, y + 26))

    def render(self):
        self.screen.fill(BG)
        self.draw_map(self.screen)
        self.draw_actors(self.screen, ghost=bool(self.proposal))
        self.draw_topbar(self.screen)
        self.draw_botbar(self.screen)
        pygame.display.flip()

    # ---- interaction ---- #
    def pick_actor(self, mx, my):
        import math
        best, bd = None, 1e9
        sc = self._live_sc()
        tau = self._tau()
        k = int(round(tau / se.DT))
        for a in sc.actors:
            kk = max(0, min(k, len(a.traj) - 1))
            px, py = w2p(*a.traj[kk][:2])
            d = math.hypot(px - mx, py - my)
            if d < bd and d < 24:
                best, bd = a.id, d
        return best

    def perturb(self, p):
        self.orch.queue_perturbation(p)
        self.orch._apply_pending()           # apply immediately at current T
        self.scrub = 0.0

    def handle_key(self, k):
        o = self.orch
        if k == pygame.K_SPACE:
            self.running_loop = not self.running_loop
        elif k == pygame.K_PERIOD and not self.running_loop:
            o.step()
        elif k == pygame.K_LEFT and not self.running_loop:
            self.scrub = max(0.0, self.scrub - 0.2)
        elif k == pygame.K_RIGHT and not self.running_loop:
            self.scrub += 0.2
        elif k == pygame.K_a:
            leg = LEGS[self.leg_i % len(LEGS)]
            self.leg_i += 1
            self.perturb(O.Perturbation("add_actor", leg=leg, turn="straight",
                                        speed=10.0))
            self.msg = f"added actor on {leg}"
        elif k == pygame.K_t and self.selected is not None:
            turn = TURNS[(self._turn_i() + 1) % len(TURNS)]
            self.perturb(O.Perturbation("set_maneuver", actor=self.selected,
                                        turn=turn))
            self.msg = f"actor {self.selected} route -> {turn}"
        elif k == pygame.K_s and self.selected is not None:
            spd = 12.0 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 3.0
            self.perturb(O.Perturbation("set_speed", actor=self.selected, speed=spd))
            self.msg = f"actor {self.selected} speed -> {spd:g}"
        elif k == pygame.K_DELETE and self.selected is not None:
            self.perturb(O.Perturbation("remove_actor", actor=self.selected))
            self.msg = f"removed actor {self.selected}"
            self.selected = None
        elif k == pygame.K_k:
            v = o.checkpoint()
            self.msg = f"checkpoint saved (v{v})"
        elif k == pygame.K_l:
            self._load_next_checkpoint()
        elif k == pygame.K_p:
            self._propose()
        elif k == pygame.K_RETURN and self.proposal:
            ver, trial, rr = self.proposal
            o.accept_proposal(ver, trial, rr)
            self.proposal = None
            self.msg = f"accepted proposal (v{ver})"

    def _turn_i(self):
        return 0

    def _propose(self):
        rr = self.orch.compute_repair()
        if rr.feasible and rr.interventions:
            trial = self.orch.build_trial(rr)
            ver = self.orch.save_proposal(rr, trial)
            self.proposal = (ver, trial, rr)
            self.msg = f"proposal v{ver}: " + "; ".join(str(i) for i in
                                                        rr.interventions)
        else:
            self.msg = "orchestrator proposes nothing (family holds or infeasible)"

    def _load_next_checkpoint(self):
        cps = [v["version"] for v in self.orch.store.versions
               if v["kind"] in ("start", "checkpoint")]
        if not cps:
            return
        cur = getattr(self, "_cp_cursor", -1)
        cur = (cur + 1) % len(cps)
        self._cp_cursor = cur
        self.orch.load_checkpoint(cps[cur])
        self.scrub = 0.0
        self.msg = f"loaded checkpoint v{cps[cur]} (new branch)"

    def loop(self):
        alive = True
        while alive:
            dt = self.clock.tick(30) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    alive = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.selected = self.pick_actor(*e.pos)
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_q,):
                        alive = False
                    elif e.key == pygame.K_ESCAPE:
                        if self.proposal:
                            self.proposal = None
                            self.msg = "proposal discarded"
                        else:
                            alive = False
                    else:
                        self.handle_key(e.key)
            if self.running_loop and not self.proposal:
                self.acc += dt
                while self.acc >= O.DT_TICK:
                    self.orch.step()
                    self.acc -= O.DT_TICK
            self.render()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(O.V2, "scenarios",
                                                   "scenario_v20.yaml"))
    ap.add_argument("--session", default=None)
    args = ap.parse_args()
    sid = args.session or (datetime.now().strftime("%Y%m%dT%H%M%S")
                           + "_interactive")
    session_dir = os.path.join(HERE, "sessions", sid)
    Editor(args.base, session_dir).loop()
    print(f"session saved: {session_dir}\n"
          f"build the tree:  python3 build_tree.py sessions/{sid}")


if __name__ == "__main__":
    main()
