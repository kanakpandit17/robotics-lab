#!/usr/bin/env bash
cd ~/Isaac-GR00T || exit 1
echo "--- dataset module exports ---"
~/.local/bin/uv run python -c "import gr00t.data.dataset as d; print([x for x in dir(d) if not x.startswith('_')])" 2>/dev/null
echo "--- dataset subpackage files ---"
ls gr00t/data/dataset/ 2>/dev/null
echo "--- modality.json examples in repo ---"
find . -name modality.json 2>/dev/null | head -3
echo "--- one modality.json content ---"
f=$(find . -name modality.json 2>/dev/null | head -1)
[ -n "$f" ] && cat "$f"
echo "--- LeRobot loader class names (grep) ---"
grep -rhoE "class Le[A-Za-z0-9_]+Dataset" gr00t/data/ 2>/dev/null | sort -u
