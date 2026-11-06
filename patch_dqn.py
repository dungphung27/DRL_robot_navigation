#!/usr/bin/env python3
"""Run in the repo root to update your EXISTING, locally modified dqn.py safely.

The script backs up dqn.py and only changes observation_space and map dimensions.
Review the diff before training.
"""
from pathlib import Path
import re
from datetime import datetime

path = Path("dqn.py")
if not path.exists():
    raise SystemExit("Run this script from the rl-robot-navigation repository root")
text = path.read_text()
changed = text
# Only patch the actual training invocation, not explanatory comments.
changed, n_dim = re.subn(r"(observation_space\s*=\s*)63\b", r"\g<1>64", changed)
changed, n_env = re.subn(r"(ObstaclesWorld\s*\(\s*)(?:500|300)\s*,\s*(?:500|300)(?=\s*,)", r"\g<1>300, 300", changed)
if n_dim != 1:
    raise SystemExit(f"Expected exactly one observation_space=63, found {n_dim}; edit manually")
if n_env != 1:
    raise SystemExit(f"Expected exactly one map constructor, found {n_env}; edit manually")
backup = path.with_name("dqn.py.backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.write_text(text)
path.write_text(changed)
print(f"Updated {path}; backed up original as {backup}")
print("Check: observation_space=64, ObstaclesWorld(300, 300, ...)")
