"""Tabulate out/<model>.json and apply the decision rule agreed before the bake-off.

    python tools/bakeoff/report.py            # any Python with only the standard library

A contender "clearly wins" only if, with the player's arms blanked:
  calm 8 fps windows: neighbours >= 85%, all pairs >= 60%, ghosts <= 10%   (about the hallway clip)
  2 fps windows:      neighbours >= 50%                                    (survives the fast turns)
"""
import json
from pathlib import Path

from common import CALM, FAST, OUT, WINDOWS

pct = lambda x: f"{100 * x:.0f}%"


def verdict(res: dict) -> tuple[bool, list[str]]:
    fails = []
    for w in CALM:
        s = res.get(w, {}).get("arms_blanked")
        if not s or s["neighbours"] < 0.85 or s["all"] < 0.60 or s["ghosts"] > 0.10:
            fails.append(f"{w}: " + (f"neighbours {pct(s['neighbours'])}, all {pct(s['all'])}, ghosts {pct(s['ghosts'])}" if s else "no result"))
    for w in FAST:
        s = res.get(w, {}).get("arms_blanked")
        if not s or s["neighbours"] < 0.50:
            fails.append(f"{w}: neighbours {pct(s['neighbours']) if s else 'n/a'}")
    return not fails, fails


def main() -> None:
    models = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
    print("Agreement between frames (arms blanked): neighbours / all pairs / ghosts\n")
    print("| window | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for name, *_ in WINDOWS:
        cells = []
        for res in models.values():
            s = res.get(name, {}).get("arms_blanked")
            cells.append(f"{pct(s['neighbours'])} / {pct(s['all'])} / {pct(s['ghosts'])}" if s else res.get(name, {}).get("error", "—"))
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    for m, res in models.items():
        ok, fails = verdict(res)
        print(f"{m}: {'CLEARLY WINS' if ok else 'does not meet the bar'}" + ("" if ok else " (" + "; ".join(fails) + ")"))


if __name__ == "__main__":
    main()
