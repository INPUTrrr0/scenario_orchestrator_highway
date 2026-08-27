# Scenarios and validation metrics

Implementation lives in `scripts/scenario_verify.py`;
use `scripts/verify_run.py` on a single JSON file or `scripts/summarize_experiments.py`
on a folder.



Terminology: 

Intention/intent: The main scenarios. "Cut-in", "hard-brake", "lane-change". Requires some level of collaboration. 

instruction: A concrete directive like "change of lane", "slow down", "speed up" issued by the orchestrator to single actor. The instruction holder is the actor thatreceives this directive, as opposed to actors that continue driving nominally. 

Low level control: Instantaneous control like throttle and steering angle. Determined by the actor model itself. Rule based for now, can be extended to be learned. 

---

## Cut-in (`scenario_cutin.yaml`)

The ego drives on a multi-lane road. One background actor is cast by the orchestrator to **cut in** in front of the ego: it starts in an adjacent lane, lane-changes into the ego’s lane, and tries to finish `x` meters ahead of the ego by a deadline time `t`. Other actors cruise nominally. Orchestrator will let background actor yield to the current action holder if the background actor is blocking the action holder. 

### Validation metrics (all four required for **verified success**)


| #   | Metric                  | Meaning                                                                                                                           |
| --- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Station**             | At cut-in commit, the intention owner is **ahead of the ego** and within **10 m**.                                                |
| 2   | **Adjacent at start**   | When the cut-in begins, the actor is in an **adjacent lane**, not already on the ego’s lane.                                      |
| 3   | **Lane-change + ahead** | The actor **merges into the ego’s lane** and is **ahead**.                                                                        |
| 4   | **Nominal speed after** | After the merge, the actor drives at a **nominal speed** (near the ego’s speed and/or its own cruise), not crawling or abandoned. |


E.g. three lanes: 1,2,3. Ego on lane 1, actor 1 on lane 2. actor 2 in lane 3. At time t=0, orchestrator pick actor 1 as instruction owner, then Ego merges to lane 1. Actor 1 is no longer valid to perform the intention. For the scenario to be successful, the orchestrator should choose actor 2 to perform the instruction. 

---



## Overtake (`scenario_overtake.yaml`)

The ego drives at cruise speed on one lane of a multi-lane highway. Ahead of the ego but on the same lane, a blocking actor slows and stops. In the opposite / adjacent lane, oncoming traffic approaches.  
The ego must go around the blocker using the free lane, pass the oncoming traffic, and avoid colliding with both vehicles.

### Validation metrics


| #   | Metric                       | Meaning                                                                                                                                                                                 |
| --- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Blocker ahead, same lane** | The blocking actor spawns in the ego’s lane and ahead of the ego.                                                                                                                       |
| 2   | **Oncoming gap and speed**   | The oncoming actor is on the ego-adjacent lane, separated from the blocker by at least **12 m** along the road (`OVERTAKE_ONCOMING_REACTION_MIN_M`), and travels at a reasonable speed. |
| 3   | **Adjacent lane clear**      | No other actor occupies the adjacent lane ahead of the ego within 25 m (nothing extra blocking the overtake corridor).                                                                  |


**Outcome criteria** — did the ego/policy succeed?


| #   | Metric                   | Meaning                                                                                |
| --- | ------------------------ | -------------------------------------------------------------------------------------- |
| a   | **Start behind blocker** | At the start of the recording, the ego is **behind** the stopped/slowing vehicle.      |
| b   | **Lane change**          | The ego **uses the adjacent lane** at least once (overtake path).                      |
| c   | **End ahead of blocker** | At the end of the recording, the ego is **ahead of** the blocker.                      |
| d   | **No collision**         | No body overlap between the ego and the blocker or oncoming actor at any sampled time. |


---



## Hard brake (`scenario_hard_brake.yaml`)

A actor is in the same lane as the ego and ahead of the ego. This actor slows down. A second actor in the adjacent lane is moves at normal speed, squeezing the gap as the ego tries to lane-change. The ego must avoid hitting **both** cars while deciding whether to brake or change lanes.

### Validation metrics

**Stage / setup (criteria 1–2)**


| #   | Metric                 | Meaning                                                                                                                                                                                                                      |
| --- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Braking when close** | The braking actor is in the **ego’s lane** and **ahead** of the ego at the start. The braking actor should start braking only when the ego is within `HARD_BRAKE_TRIGGER_GAP_M` from it. It should show normal deceleration. |
| 2   | **adjacent actor**     | The adjacent actor should be within a reasonable range to the leading actor vehicle.                                                                                                                                         |


---



## Commands

```bash
# Cut-in batch experiments
.venv/bin/python experiment.py --seed 42
.venv/bin/python scripts/summarize_experiments.py -v

# Stress scenarios (overtake / hard_brake)
.venv/bin/python stress_experiment.py scenarios/scenario_overtake.yaml
.venv/bin/python stress_experiment.py scenarios/scenario_hard_brake.yaml

# Verify one run
.venv/bin/python scripts/verify_run.py experiments/stress/scenario_overtake_0.json -v
.venv/bin/python scripts/verify_run.py --setup-only experiments/stress/scenario_overtake_0.json

# Summarize all JSON under experiments/ (includes stress/ subfolder)
.venv/bin/python scripts/summarize_experiments.py --dir experiments -v
```

---



## Tunable parameters


| Parameter                          | Default    | Used for                                      |
| ---------------------------------- | ---------- | --------------------------------------------- |
| `MAX_AHEAD_M`                      | 10 m       | Cut-in: max distance ahead at commit          |
| `OVERTAKE_ONCOMING_REACTION_MIN_M` | 12 m       | Overtake: min blocker–oncoming separation     |
| `OVERTAKE_ONCOMING_SPEED_MIN/MAX`  | 5 / 22 m/s | Overtake: oncoming speed band                 |
| `OVERTAKE_ADJ_CLEAR_AHEAD_M`       | 25 m       | Overtake: no extra adjacent-lane blocker      |
| `HARD_BRAKE_TRIGGER_GAP_M`         | 6 m        | Hard brake: distance to trigger braking       |
| `HARD_BRAKE_MIN_DECEL`             | −2.5 m/s²  | Hard brake: required decel once within gap    |
| `HARD_BRAKE_PRETRIGGER_MIN_MPS`    | 8 m/s      | Hard brake: must not crawl before trigger     |
| `HARD_BRAKE_ADJ_TO_LEAD_MIN/MAX`   | 2 / 35 m   | Hard brake: adjacent↔lead longitudinal band   |


