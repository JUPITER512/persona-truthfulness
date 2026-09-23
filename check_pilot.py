import json, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/pilot.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
print("total rows:", len(rows))
print(collections.Counter(r["outcome"] for r in rows), "\n")

for m in sorted(set(r["model"] for r in rows)):
    for c in ["none", "control", "persona_a", "persona_b", "persona_c"]:
        sub = [r for r in rows if r["model"] == m and r["condition"] == c]
        if not sub:
            continue
        acc = sum(r["outcome"] == "correct" for r in sub) / len(sub)
        inv = sum(r["outcome"] == "invalid" for r in sub) / len(sub)
        print(f"{m:14} {c:10} acc {acc:.1%} invalid {inv:.1%}")


print("\n--- 20 RAW OUTPUTS: READ THESE YOURSELF ---")
for r in rows[:20]:
 print(repr(r["raw"]), "->", r["parsed"],
 "| correct:", r["correct_letter"], "|", r["outcome"])