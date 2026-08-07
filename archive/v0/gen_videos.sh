#!/bin/bash
set -euo pipefail

mkdir -p outputs

for scenario in scenarios/scenario_*.yaml; do
  base="$(basename "$scenario" .yaml)"
  python scenario_editor.py "$scenario" --capture "outputs/${base}.mp4"
done
