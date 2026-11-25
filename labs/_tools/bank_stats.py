"""Statistics of the practice bank and the deck checkpoints: key positions, true/false balance and
length cues (is the key the only longest or only shortest option?).

    cd labs
    uv run python _tools/bank_stats.py
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def report(name, items):
    single = [i for i in items if len(i[0]) > 2]
    tf = [i for i in items if len(i[0]) == 2]
    pos = Counter("ABCDEF"[k] for _, k in single)
    longest = sum(len(o[k]) == max(map(len, o)) and [len(x) for x in o].count(len(o[k])) == 1 for o, k in single)
    shortest = sum(len(o[k]) == min(map(len, o)) and [len(x) for x in o].count(len(o[k])) == 1 for o, k in single)
    print(f"{name}: {len(items)} items; key positions {dict(sorted(pos.items()))}; "
          f"key the only longest in {longest}, the only shortest in {shortest} of {len(single)}"
          + (f"; true/false keys {dict(Counter(o[k] for o, k in tf))}" if tf else ""))


bank = json.loads((ROOT / "assessments" / "bank.json").read_text(encoding="utf-8"))["questions"]
report("bank", [(q["options"], q["correct"][0]) for q in bank])
print("bank by outcome:", dict(sorted(Counter(q["lo"][0] for q in bank).items())))

quizzes = []
for deck in sorted((ROOT / "slides").glob("week*.html")):
    for block in re.findall(r'<div class="opts">(.*?)</div>\s*<div class="callout', deck.read_text(encoding="utf-8"), re.S):
        opts = re.findall(r'<div class="opt"([^>]*)>(.*?)</div>', block, re.S)
        texts = [html.unescape(re.sub(r"<[^>]+>", "", t)).strip() for _, t in opts]
        key = [i for i, (attr, _) in enumerate(opts) if "data-ok" in attr]
        if key:
            quizzes.append((texts, key[0]))
report("deck checkpoints", quizzes)
