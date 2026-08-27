# Highway maps for the CARLA port

Which road each scenario should run on, and why. Regenerate the survey tables
with `python3 -m carla_highway.survey_maps`; write the generated maps with
`python3 -m carla_highway.make_maps`.

---

## TL;DR

| I want | Use |
|---|---|
| A clean recording of any scenario | `--xodr auto` (generated straight highway, no scenery) |
| 2 same-direction lanes | `maps/highway_2lane.xodr`, or **Town06 road 40** |
| 3 same-direction lanes | `maps/highway_3lane.xodr`, or **Town06 road 40** |
| 4 same-direction lanes | `maps/highway_4lane.xodr`, or **Town06 road 40** |
| 2 lanes, opposing traffic (`overtake`) | `maps/highway_2lane_twoway.xodr`, or **Town01 road 8** |

```bash
python3 -m carla_highway.make_maps                                  # once
python3 -m carla_highway.runner --scenario cutin      --xodr auto
python3 -m carla_highway.runner --scenario hard_brake --xodr auto
python3 -m carla_highway.runner --scenario overtake   --xodr auto
```

---

## 1. Do the stock towns have a real highway?

Yes — one. **Town06 road 40, section 0: 470 m of junction-free four-lane
straight**, 3.50 m lanes. It is the only section in any shipped town that
carries four same-direction lanes for more than 230 m without an intersection.

"Junction-free" is the criterion that matters and it is not the default.
`HighwayFrame.discover` measures its straight run with
`_straight_run(through_junctions=True)`, which walks *through* intersections by
taking the branch that best continues the heading. That is right for getting a
scenario to run anywhere, and it is why `overtake` originally fitted Town04
road 3 — a genuinely 170 m straight that happens to thread a town centre. The
manoeuvre was correct and nearly impossible to watch behind buildings, parked
cars and cross traffic.

The survey below measures with `through_junctions=False`: every metre reported
is road the scenario can use without crossing traffic.

### Same-direction lanes, junction-free

| Town | Best section | Junction-free run | Lanes available |
|---|---|---|---|
| **Town06** | **road 40 sec 0** | **470 m** | 2 / 3 / 4 |
| Town06 | road 29, road 37 | 190 m | 2 / 3 / 4 |
| Town04 | road 40 sec 0 | 230 m | 2 / 3 / 4 |
| Town04 | road 38 | 190 m | 2 / 3 / 4 |
| Town03 | road 3 | 110 m | 2 only |
| Town05 | road 44 | 60 m | 2 only |
| Town01, Town02, Town07, Town10HD | — | none | — |

Town06 is CARLA's highway map and this is what that means concretely: 17
qualifying two-lane sections against Town04's 6, and the only 470 m run in the
set.

### Two-way (what `overtake` needs), junction-free

| Town | Best section | Junction-free run | Lane width |
|---|---|---|---|
| **Town01** | **road 8 sec 0** | **310 m** | 4.00 m |
| Town01 | road 15 | 300 m | 4.00 m |
| Town02 | road 12 | 170 m | 4.00 m |
| Town04 | road 39 | 130 m | **10.50 m** ← divided highway, see below |
| Town07 | road 58 | 100 m | 3.20 m |
| Town06 | road 54 | 60 m | 8.00 m |

Two cautions here. Town04's two-way candidates report a **10.50 m lane width**:
those are divided highways where the two carriageways are separated by a
median, and the "lane width" the fitter derives is the gap across it. The
scenario's actors are placed on an evenly spaced grid
(`MapConfig.lane_center_x`), so a 10.5 m spacing puts the oncoming car three
lanes away rather than one — the scenario still runs, but it is not the
scenario that was authored. Town01 road 8 at 4.00 m is the honest stock choice,
and it is a town street, not a highway.

---

## 2. The generated maps

`carla_highway/make_maps.py` writes four OpenDRIVE files that
`client.generate_opendrive_world()` turns into a drivable world with, in the
CARLA API's own words, *"no graphics besides the road and sidewalks"*.

| File | Lanes | Length | For |
|---|---|---|---|
| `maps/highway_2lane.xodr` | 2 one-way | 600 m | `hard_brake` |
| `maps/highway_3lane.xodr` | 3 one-way | 600 m | `cutin` |
| `maps/highway_4lane.xodr` | 4 one-way | 600 m | headroom |
| `maps/highway_2lane_twoway.xodr` | 1 + 1 | 600 m | `overtake` |

All are straight, flat, 3.50 m lanes, with a 2 m shoulder each side. The fit
error against the script's straight frame is **0.000 m** — it cannot be
anything else, since the road *is* a straight line.

They exist for two reasons, and only the second is about aesthetics:

1. **`overtake` has no good stock road.** The best junction-free two-way
   straight in any shipped town is 310 m of Town01 town street; every highway
   with opposing traffic is divided, so its "adjacent" lane is a median away.
2. **Nothing occludes the manoeuvre.** No buildings, no parked cars, no cross
   traffic, no scenery. A 600 m road is also longer than any stock straight, so
   the actors' 40 s scripts never run off the end of the fit.

The scenarios are unchanged and the results are unchanged: all four grade
identically on the generated maps and on Town04, which is the point — the map
is a stage, not a variable.

### Registering a map for a scenario

`ModeSpec.xodr` names the map each mode is meant to be recorded on, which is
what `--xodr auto` resolves. Adding a mode means adding its map there.

### A wrinkle worth knowing

`generate_opendrive_world` writes the OpenDRIVE it was handed back into the
CARLA install's `Content/Carla/Maps/OpenDrive/` as **`OpenDriveMap.xodr`**.
After any run on a generated map that file is on disk, and a naive survey
reports our own road as a shipped town. `survey_maps.GENERATED_WRITEBACK`
filters it out.

---

## 3. A fit bug the survey exposed

`_candidate_sections` used to return the **first** waypoint it saw per road
section. `_fit_section` measures its run forwards *and* backwards from that
seed and keeps `2 * min(fwd, bwd)`, so a seed at either end of a section scores
zero however long the road is.

On the stock towns this quietly truncated fits — the first survey reported
Town06's best as 300 m when it is 470 m, and found one qualifying section where
there are seventeen. On a generated single-road map it is fatal: the seed lands
at `s = 0` and the map reports as having no straight in it at all.

Seeds are now taken from the **middle** of each section.

---

## 4. Reproducing

```bash
# the survey tables above
python3 -m carla_highway.survey_maps                      # same-direction
python3 -m carla_highway.survey_maps --two-way --lanes 2  # overtake's road
python3 -m carla_highway.survey_maps --allow-junctions    # what discover() sees

# the generated maps
python3 -m carla_highway.make_maps                        # -> maps/
python3 -m carla_highway.make_maps --length 1200 --only highway_4lane
```
