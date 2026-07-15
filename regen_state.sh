#!/usr/bin/env bash
#
# regen_state.sh — regenerate the persisted scenario-editor state so it stays
# consistent with the CURRENT scenario format/representation.
#
# Run this whenever the editor's format changes (or after hand-editing files).
# It is safe and idempotent. It does three things:
#   1. Resets the append-only edit history (edit_history.yaml -> empty).
#   2. Validates every scenario_v*.yaml against the current format and moves any
#      that no longer parse into scenarios/incompatible/ (nothing is deleted).
#   3. Rebuilds provenance.yaml so it only references files that still exist and
#      validate, preserving parent links where the parent file is still present.
#
# Usage:  ./regen_state.sh [SCENARIOS_DIR]
#         (default SCENARIOS_DIR is ./scenarios next to this script)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:-$HERE/scenarios}"
EDITOR="$HERE/scenario_editor.py"

echo "Scenarios dir: $DIR"
[ -d "$DIR" ] || { echo "no such directory: $DIR" >&2; exit 1; }

# 1) reset the edit history (it is a regenerable log, not source of truth)
: > "$DIR/edit_history.yaml"
echo "  reset edit_history.yaml"

# 2) validate scenario files; quarantine incompatible ones (move, never delete)
shopt -s nullglob
for f in "$DIR"/scenario_v*.yaml; do
  if python3 "$EDITOR" --validate "$f" >/dev/null 2>&1; then
    echo "  valid:        $(basename "$f")"
  else
    mkdir -p "$DIR/incompatible"
    echo "  INCOMPATIBLE: $(basename "$f") -> incompatible/"
    mv "$f" "$DIR/incompatible/$(basename "$f")"
  fi
done
shopt -u nullglob

# 3) rebuild provenance.yaml from the files that remain
python3 - "$DIR" <<'PY'
import os, re, sys, yaml
d = sys.argv[1]
prov = os.path.join(d, "provenance.yaml")

old = {}
if os.path.exists(prov):
    for v in (yaml.safe_load(open(prov)) or {}).get("versions", []):
        old[v.get("file")] = v

def vnum(fn):
    m = re.search(r"scenario_v(\d+)\.yaml$", fn)
    return int(m.group(1)) if m else None

present = sorted(f for f in os.listdir(d)
                 if f.startswith("scenario_v") and f.endswith(".yaml"))
present_nums = {vnum(f) for f in present}

versions = []
for f in present:
    e = old.get(f, {})
    parent = e.get("parent")
    if parent is not None and parent not in present_nums:
        parent = None  # parent file gone -> becomes a root
    versions.append({"version": vnum(f), "file": f, "parent": parent,
                     "created": e.get("created", "regenerated")})
versions.sort(key=lambda v: v["version"])

with open(prov, "w") as fh:
    yaml.safe_dump({"versions": versions}, fh, sort_keys=False)
print(f"  rebuilt provenance.yaml ({len(versions)} version(s))")
PY

echo "done."
