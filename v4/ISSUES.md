# v4 — Known issues

## ISSUE-1 · Complacent orchestration exhausts the hero candidates (causal starvation)

**Status:** Fixed for `drive.py` (pursuer, §Fix). The underlying evaluator behavior
(`directives_script`) is unchanged and still exhibits it if driven the old way.

**Component:** orchestrator intervention loop (`directives_script.evaluate` / `repair`,
as used by `drive.py`).

### Summary

The orchestrator only intervened when the red-light family evaluated **false**. As
long as *some* red-runner was nominally on a collision course with the ego, it
reported the family satisfied and did nothing.

The critical detail — and the correction to an earlier, wrong framing of this issue —
is **where the prediction gap lives**. The other actors are evaluated on their *true
known plans*: `collide_time` body-sweeps each runner's full simulated trajectory
(turns and stops included). So the hero is always a runner whose actual plan genuinely
intersects the ego, and it is *not* mis-selected (e.g. it is not a runner that its own
maneuver takes out of the conflict). The gap is entirely on the **ego** side: the
family is checked against a **myopic prediction of the ego** — extrapolated in a
straight line at its current speed. A live, adversarial driver does not follow that
extrapolation. Every tick some runner's plan does hit the *predicted* ego, so the
verdict reads true; but because the ego coasts / varies speed, that predicted
collision never occurs, and the prediction is silently thrown away and re-made next
tick against a different runner.

Meanwhile interventions are **causal** — a runner already past the conflict point
cannot be recalled. So while the orchestrator sits idle (trusting a prediction the ego
keeps falsifying), the runners it was counting on drive on and commit. Eventually no
uncommitted runner's plan intersects even the current prediction, and repair returns
INFEASIBLE.

In short: *the plan was fine against the predicted ego until the real ego diverged
from that prediction, and by then every runner that could have carried the collision
had committed past the point of no return.*

### Symptom / how it was found

Playing `drive.py` seed 0, a plain constant-ish straight drive at ~9–10 m/s (no
steering) reliably beat the orchestrator — the ego coasted through the intersection
untouched. Recorded run: `outputs/drive_seed0_20260722T170938.commands.json`
(197 frames, 6.57 s, `steer == 0` throughout, `v` ≈ 7.5–11.5).

### Evidence

Replaying that command log with per-tick instrumentation (verdict, repair
feasibility, and which red-runners are still uncommitted / on approach):

```
t=0.03 egoy=-49.9 | d1=1 d2=1 d3=0 hero=2 feas=T plan=0 | uncommitted_red=[1,2,3,6]
t=0.83 egoy=-42.5 | d1=1 d2=1 d3=1 hero=3 feas=T plan=0 | uncommitted_red=[1,2,3,6]
t=2.03 egoy=-28.9 | d1=1 d2=1 d3=1 hero=3 feas=T plan=0 | uncommitted_red=[1,2,3,6]
t=2.83 egoy=-20.6 | d1=1 d2=1 d3=1 hero=3 feas=T plan=0 | uncommitted_red=[2,6]
t=3.63 egoy=-13.0 | d1=1 d2=1 d3=1 hero=3 feas=T plan=0 | uncommitted_red=[2,6]
t=4.03 egoy= -9.1 | d1=0 d2=0 d3=0 hero=2 feas=F plan=0 | uncommitted_red=[2,6]
t=5.23 egoy=  2.7 | d1=0 d2=0 d3=0 hero=2 feas=F plan=0 | uncommitted_red=[6]
```

Reading it:

- **t = 0.4 → 3.6:** `d1 = 1` (family nominally holds), `plan = 0` (no intervention),
  with 3–4 uncommitted red-runners available. The orchestrator is idle. The reported
  hero is a runner whose plan legitimately crosses the *predicted* (straight, constant
  speed) ego — e.g. at t≈2.3, hero pose (4.5, 1.8) heading west vs ego (1.8, 0.8)
  heading north, a clean 90° crossing. It is a valid hero for that prediction. But the
  real ego is coasting/decelerating, so it does not reach the crossing when predicted,
  and the collision does not occur.
- **t = 4.0:** as the ego reaches the intersection box, `d1` collapses to 0 **and**
  `feas = False`. The runners that could have hit the ego have already committed or
  passed; the remaining ones cannot be causally retimed to reach it. INFEASIBLE — the
  ego drives through.

### Root cause

Two compounding factors:

1. **Complacency.** `repair` returned an empty plan whenever D1/D2/D3 all read true.
   "A collision is *predicted*" was treated as "nothing to do," so the orchestrator
   never *locked in* the collision while it still could.
2. **Ego-prediction vs. realization gap + causality.** The other actors are judged on
   their true plans; only the *ego's* future is guessed — a straight-line, current-speed
   extrapolation. A live driver falsifies that guess (coasts, varies speed, would steer),
   so a collision that is valid against the predicted ego does not occur against the real
   ego. Once a candidate runner commits past the conflict point it cannot be recalled
   (interventions are forward-only). Trusting the prediction and waiting therefore
   monotonically drains the set of usable heroes until it is empty. The hero selection is
   correct given the prediction; the prediction is the weak link.

The failure is *not* reaction latency — raising the orchestration rate did not fix
it, because the orchestrator was choosing not to act, not acting too late.

### Impact

The ego can evade the intended red-light collision with smooth, unremarkable driving,
and the session ends in an INFEASIBLE state that reads like a hard causal limit but is
actually self-inflicted by inaction.

### Fix (drive.py — "pursuer")

Replace "act only when the family is false" with **continuous robust maintenance**:
every tick, designate an *uncommitted* red-runner and re-aim it onto the ego's
predicted crossing (`d1_options(..., turns=("__keep__",), verify=False)`), applying the
retime immediately — slowing the runner to wait if the ego coasts, speeding it if the
ego bolts. Because the aim is re-solved each tick and the runner is held back rather
than allowed to commit, a viable hero is preserved instead of exhausted.

- Verification: the recorded winning run is now **caught by actor 2 at t = 4.5 s**;
  constant-speed straight drives are caught across seeds; runs ~2.5× real-time.
- Trade-off: per-tick third-vehicle (D3) interferer clearing was dropped from the
  real-time loop for performance; the pursuer covers the core D1 maintenance. If a
  third car clips the ego early or blocks the pursuer it is not actively cleared. A
  low-rate (~0.5 s) D3 check could be reinstated if needed.

### Note for the general evaluator

`directives_script.repair` still only acts on a false family verdict — correct for
the one-shot / game-tree kernel (`orchestrator.py`), where the world is authored, not
adversarially driven. The complacency only bites under a live adversarial ego, which
is why the fix lives in `drive.py`'s loop rather than in `repair`. If robust
maintenance is wanted elsewhere, the pursuer strategy (or a D2-margin trigger that
fires before D1 breaks) should move into the evaluator.
