"""I(FM;Y) and FM-identity-only classifier accuracy per scheme, 30 seeds.
Pure bookkeeping on assignments + slide counts; no training."""
import json, sys, math
from collections import Counter
sys.path.insert(0, '<repo-root>')
from hetfm import assign
import numpy as np
SEEDS = list(range(62, 92))
SCHEMES = ["balanced", "stratified", "cancer_correlated", "adversarial"]
out = {s: [] for s in SCHEMES}
for sd in SEEDS:
    _, sp, gt, _ = assign.build_canonical_partition(seed=sd)
    clab = assign.client_labels_from_splits(sp)
    # slide-level (train) label per client
    ntrain = {c: len(sp[c]["train"]) for c in sp}
    ntest = {c: len(sp[c]["test"]) for c in sp}
    for sc in SCHEMES:
        asn = assign.assign_fms(list(sp), sc, sd, client_label=clab)
        # joint counts over training slides
        J = Counter()
        for c in sp:
            J[(asn[c], clab[c])] += ntrain[c]
        N = sum(J.values())
        pf = Counter(); py = Counter()
        for (f, y), n in J.items():
            pf[f] += n; py[y] += n
        mi = sum(n / N * math.log2((n / N) / ((pf[f] / N) * (py[y] / N))) for (f, y), n in J.items())
        hy = -sum(n / N * math.log2(n / N) for n in py.values())
        hf = -sum(n / N * math.log2(n / N) for n in pf.values())
        # FM-only classifier: predict majority class of the FM group (fit on train), score macro-acc on test slides
        maj = {}
        for f in pf:
            maj[f] = max(py, key=lambda y: J[(f, y)])
        yt, yp = [], []
        for c in sp:
            for s in sp[c]["test"]:
                yt.append(clab[c]); yp.append(maj[asn[c]])
        yt = np.array(yt); yp = np.array(yp)
        accs = [float((yp[yt == y] == y).mean()) for y in sorted(set(yt))]
        macro = float(np.mean(accs)); overall = float((yt == yp).mean())
        n_classes_per_fm = {f: sum(1 for (ff, y) in J if ff == f) for f in pf}
        out[sc].append({"seed": sd, "I_FM_Y_bits": round(mi, 4), "H_Y_bits": round(hy, 4), "H_FM_bits": round(hf, 4),
                        "normalized_I_over_HY": round(mi / hy, 4), "fm_only_macro_acc": round(macro, 4),
                        "fm_only_overall_acc": round(overall, 4), "classes_seen_per_fm": n_classes_per_fm})
    print("seed", sd, {sc: out[sc][-1]["I_FM_Y_bits"] for sc in SCHEMES}, flush=True)
summ = {}
for sc in SCHEMES:
    for key in ["I_FM_Y_bits", "normalized_I_over_HY", "fm_only_macro_acc", "fm_only_overall_acc"]:
        xs = [r[key] for r in out[sc]]
        summ.setdefault(sc, {})[key] = {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4), "min": min(xs), "max": max(xs)}
res = {"seeds": SEEDS, "per_seed": out, "summary": summ,
       "note": "I(FM;Y) over training slides (FM = which foundation model the slide's site holds; Y = 9-class label). fm_only = classifier that predicts the majority training class of the slide's FM group; macro accuracy on the global test set."}
json.dump(res, open('<out-dir>/mi_fm_y.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
