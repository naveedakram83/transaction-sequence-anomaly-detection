"""Point vs sequence-aware anomaly detection on synthetic card transactions.
Reproducible: python experiment.py  (seed fixed)."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(42)
N_CARDS, DAYS = 2000, 30
MCCS = np.arange(20)

def normal_stream(card):
    mu = rng.normal(3.5, 0.6)                      # card's typical log-amount
    fav = rng.choice(MCCS, size=4, replace=False)  # card's usual merchant categories
    n = rng.poisson(2.0 * DAYS)
    t = np.sort(rng.uniform(0, DAYS * 86400, n))
    hour = (t % 86400) / 3600
    keep = rng.random(n) < np.where((hour > 7) & (hour < 23), 1.0, 0.15)  # mostly daytime
    t = t[keep]; n = len(t)
    return pd.DataFrame({"card": card, "ts": t,
                         "amount": np.round(np.exp(rng.normal(mu, 0.5, n)), 2),
                         "mcc": rng.choice(fav, n), "label": 0, "kind": "normal"})

def inject(df, card):
    """Sequence anomalies: each looks ordinary as a single transaction."""
    kind = rng.choice(["card_testing", "takeover", "velocity"])
    base = rng.uniform(5 * 86400, (DAYS - 1) * 86400)
    hist = df[df.card == card]
    typical = hist.amount.median()
    if kind == "card_testing":      # burst of small, normal-looking amounts
        k = rng.integers(6, 12)
        ts = base + np.cumsum(rng.uniform(10, 60, k))
        amt = np.round(rng.uniform(1, 15, k), 2); mcc = rng.choice(MCCS, k)
    elif kind == "takeover":        # small test, then amounts near typical in new categories
        k = 4
        ts = base + np.array([0, 300, 900, 1500])
        amt = np.round([rng.uniform(1, 5)] + list(typical * rng.uniform(1.5, 3, 3)), 2)
        new = np.setdiff1d(MCCS, hist.mcc.unique()); mcc = rng.choice(new, k)
    else:                           # rapid repeats at typical amount, same category
        k = rng.integers(5, 9)
        ts = base + np.cumsum(rng.uniform(30, 120, k))
        amt = np.round(typical * rng.uniform(0.8, 1.2, k), 2); mcc = np.repeat(hist.mcc.mode()[0], k)
    return pd.DataFrame({"card": card, "ts": ts, "amount": amt, "mcc": mcc, "label": 1, "kind": kind})

df = pd.concat([normal_stream(c) for c in range(N_CARDS)], ignore_index=True)
bad_cards = rng.choice(N_CARDS, 100, replace=False)
df = pd.concat([df] + [inject(df, c) for c in bad_cards], ignore_index=True)
df = df.sort_values(["card", "ts"]).reset_index(drop=True)

# ---- point features: what a single transaction looks like in isolation
df["hour"] = (df.ts % 86400) / 3600
df["log_amt"] = np.log1p(df.amount)
point_feats = ["log_amt", "hour", "mcc"]

# ---- sequence features: computed only from each card's PAST (no leakage)
g = df.groupby("card")
df["dt_prev"] = np.log1p(g.ts.diff().fillna(86400 * 7))
hist_mean = g.log_amt.transform(lambda s: s.shift().expanding().mean()).fillna(df.log_amt)
hist_std = g.log_amt.transform(lambda s: s.shift().expanding().std()).fillna(1).clip(lower=0.1)
df["amt_z"] = (df.log_amt - hist_mean) / hist_std
def count_window(s, w=600):
    v = s.values; return np.array([((v[i] - v[:i]) <= w).sum() for i in range(len(v))])
df["n_10min"] = g.ts.transform(count_window)
df["new_mcc"] = (g.mcc.transform(lambda s: [int(x not in set(s.iloc[:i])) for i, x in enumerate(s)])).astype(int)
df.loc[g.cumcount() < 5, "new_mcc"] = 0  # cold start: too little history
seq_feats = point_feats + ["dt_prev", "amt_z", "n_10min", "new_mcc"]

def score(feats):
    m = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(df[feats])
    return -m.score_samples(df[feats])

res = {}
for name, feats in [("Point features", point_feats), ("Sequence-aware", seq_feats)]:
    s = score(feats); df[name] = s
    k = int(df.label.sum())
    top = df.nlargest(k, name)
    res[name] = {"PR-AUC": average_precision_score(df.label, s), "ROC-AUC": roc_auc_score(df.label, s),
                 "Precision@k": top.label.mean(),
                 **{f"recall_{t}": top[top.kind == t].shape[0] / (df.kind == t).sum()
                    for t in ["card_testing", "takeover", "velocity"]}}

print(f"transactions={len(df)}, anomalies={int(df.label.sum())} ({df.label.mean():.2%})")
print(pd.DataFrame(res).round(3).to_string())
df.to_csv("results.csv", index=False)
