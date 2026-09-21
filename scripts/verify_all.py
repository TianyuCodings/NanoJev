#!/usr/bin/env python3
"""One-command independent verification of NanoJev's published claims.

Verifies (from this repository's own bytes, no network, no dependencies):
  A. README three-system results vs web/side_by_side_results.json summaries
  B. SOURCE_MANIFEST.json per-file sha256 integrity
  C. Calibration benchmark: summary vs runs (5 arms x 3 seeds recomputed means)
  D. Calibration gradient-check source hashes (in-file `sources` claims)

Usage:  python3 scripts/verify_all.py   (from repo root)
Exit code 0 = all checks passed; 1 = any FAIL; 2 = environment problem.

Tip: run on a clone made with `git clone --config core.autocrlf=false`
(line-ending conversion breaks byte hashes — check B detects and says so).

Contributed by Nautilus Assay (independent verifier). Receipts for these
claims (signed, ed25519) live at:
https://github.com/chunxiaoxx/nautilus-compass/tree/main/docs/wall
Optional: `pip install assay-verify` to also verify those receipts locally.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Claims as printed in README (source of truth for this table = README tables;
# numbers below transcribed at contribution time — recheck against README).
CLAIMS = {
    # game -> system -> {field: value}
    "maze": {
        "NanoJev": {"steps": 244, "collisions": 36},
        "Jev": {"steps": 2738, "collisions": 1044},
        "Untuned Qwen": {"steps": 4726, "collisions": 2044},
    },
    "snake": {
        "NanoJev": {"score": 27},
        "Jev": {"score": 30},
        "Untuned Qwen": {"score": 25},
    },
}

ARMS = ["initial", "observed_ce", "direct_brier", "paired_brier_pg", "correctness_reinforce"]
METRICS = ["observed_brier", "top_label_ece", "accuracy"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_a() -> tuple[str, str]:
    """README claims vs side_by_side_results.json."""
    src = ROOT / "web" / "side_by_side_results.json"
    if not src.is_file():
        return "SKIP", "web/side_by_side_results.json missing"
    data = json.loads(src.read_text(encoding="utf-8"))
    games = {ex["game"]: ex for ex in data.get("examples", [])}
    fails = []
    checked = 0
    for game, systems in CLAIMS.items():
        ex = games.get(game)
        if not ex:
            fails.append(f"{game}: no example in json")
            continue
        by_name = {s["name"]: s.get("summary", {}) for s in ex.get("systems", [])}
        alias = {"NanoJev": "NanoJev", "Jev": "Jev", "Untuned Qwen": "Untuned Qwen"}
        for sysname, fields in systems.items():
            summ = by_name.get(alias[sysname])
            if summ is None:
                # try substring match (system naming drift)
                cand = [v for k, v in by_name.items() if sysname.lower() in k.lower()]
                summ = cand[0] if cand else None
            if summ is None:
                fails.append(f"{game}/{sysname}: system missing in json")
                continue
            for field, claimed in fields.items():
                checked += 1
                got = summ.get(field)
                if got != claimed:
                    fails.append(f"{game}/{sysname}.{field}: claimed {claimed}, json {got}")
    if fails:
        return "FAIL", f"{len(fails)} mismatch(es): " + "; ".join(fails[:4])
    return "PASS", f"{checked} claim values match web/side_by_side_results.json"


def check_b() -> tuple[str, str]:
    """SOURCE_MANIFEST.json per-file sha256."""
    mf = ROOT / "SOURCE_MANIFEST.json"
    if not mf.is_file():
        return "SKIP", "SOURCE_MANIFEST.json missing"
    man = json.loads(mf.read_text(encoding="utf-8"))
    entries = man.get("files") or man.get("entries") or man
    if isinstance(entries, dict):
        items = list(entries.items()) if not all(isinstance(v, dict) for v in entries.values()) \
            else [(v.get("path", k), v.get("sha256")) for k, v in entries.items()]
    elif isinstance(entries, list):
        items = [(e.get("path"), e.get("sha256")) for e in entries]
    else:
        return "SKIP", f"unrecognized manifest shape: {type(entries).__name__}"
    ok = bad = missing = 0
    first_bad = []
    for path, expect in items:
        if not expect:
            continue
        f = ROOT / str(path)
        if not f.is_file():
            missing += 1
            continue
        if sha256_file(f) == expect:
            ok += 1
        else:
            bad += 1
            if len(first_bad) < 3:
                first_bad.append(str(path))
    total = ok + bad + missing
    if bad or missing:
        hint = " (CRLF? re-clone with core.autocrlf=false)" if bad else ""
        return "FAIL", f"{bad} hash mismatch, {missing} missing of {total}{hint}: {first_bad}"
    return "PASS", f"{ok}/{total} file hashes match SOURCE_MANIFEST.json"


def check_c() -> tuple[str, str]:
    """Calibration benchmark summary vs recomputed means over runs."""
    src = ROOT / "results" / "calibrated_learning_benchmark.json"
    if not src.is_file():
        return "SKIP", "results/calibrated_learning_benchmark.json missing"
    d = json.loads(src.read_text(encoding="utf-8"))
    runs = d.get("runs") or []
    if not runs:
        return "SKIP", "no runs array"
    fails, checked = [], 0
    for arm in ARMS:
        for m in METRICS:
            try:
                claimed = d["summary"][arm]["test"][m]["mean"]
                vals = [r[arm]["test"][m] for r in runs if arm in r]
            except (KeyError, TypeError):
                continue
            if not vals:
                continue
            checked += 1
            got = sum(vals) / len(vals)
            if abs(got - claimed) > 1e-9:
                fails.append(f"{arm}.{m}: summary {claimed:.6f} vs recomputed {got:.6f}")
    if fails:
        return "FAIL", f"{len(fails)} aggregation mismatch(es): " + "; ".join(fails[:3])
    return "PASS", f"{checked} summary metrics = recomputed means over {len(runs)} runs"


def check_d() -> tuple[str, str]:
    """Gradient-check file claims its own sources' sha256 — recompute."""
    src = ROOT / "results" / "calibrated_objectives_check.json"
    if not src.is_file():
        return "SKIP", "results/calibrated_objectives_check.json missing"
    d = json.loads(src.read_text(encoding="utf-8"))
    sources = d.get("sources") or {}
    pairs = [("test", ROOT / "scripts" / "test_calibrated_objectives.py"),
             ("objectives", ROOT / "scripts" / "calibrated_objectives.py")]
    fails, ok = [], 0
    for key, f in pairs:
        expect = sources.get(key)
        if not expect:
            continue
        if not f.is_file():
            fails.append(f"{key}: {f.name} missing")
            continue
        if sha256_file(f) == expect:
            ok += 1
        else:
            fails.append(f"{key}: hash drift (CRLF? see header tip)")
    if fails:
        return "FAIL", "; ".join(fails)
    return "PASS", f"{ok} source hashes match in-file claims"


def main() -> int:
    print("NanoJev independent claim verification (contributed by Nautilus Assay)")
    print("=" * 72)
    results = [check_a, check_b, check_c, check_d]
    allpass = True
    for fn in results:
        label = fn.__name__.replace("check_", "").upper()
        try:
            status, detail = fn()
        except Exception as e:  # malformed file etc.
            status, detail = "FAIL", f"error: {e}"
        print(f"[{status:4}] {label}: {detail}")
        if status != "PASS":
            allpass = False
    print("=" * 72)
    print("VERDICT:", "ALL PASS" if allpass else "SEE ABOVE")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
