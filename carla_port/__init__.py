"""carla_port — a thin CARLA 0.9.16 backend for the existing maneuver-script
orchestrator (v4/orchestrator.py + v4/directives_script.py).

Nothing in this package implements orchestration semantics. D1/D2/D3/D4,
repair(), retiming/rerouting, intervention ordering and costs all stay in
v4/directives_script.py; the closed loop stays in v4/orchestrator.py. This
package supplies the world around them: map frame, actor bindings, traffic
lights, state synchronization, collision sensors and the run loop.
"""
