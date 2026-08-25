"""Extract summary metrics from all benchmark JSON results."""
import json
import os

results_dir = os.path.join(os.path.dirname(__file__), "results")

for fname in sorted(os.listdir(results_dir)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(results_dir, fname)) as f:
        data = json.load(f)
    kind = data["kind"]
    emb = data.get("embedder_name", "?")
    gran = data.get("granularity", "?")
    print(f"=== {fname} ===")
    print(f"  Kind: {kind}  Embedder: {emb}  Granularity: {gran}")
    for run in data.get("runs", []):
        em = run.get("external_metrics", {})
        im = run.get("intrinsic_metrics", {})
        tier = run["tier_actual"]
        n = run["n_docs"]
        pk = run["n_pred_clusters"]
        tc = run["n_true_classes"]
        ari = em.get("ari", 0)
        nmi = em.get("nmi", 0)
        vm = em.get("v_measure", 0)
        homo = em.get("homogeneity", 0)
        comp = em.get("completeness", 0)
        cov = em.get("coverage", 0)
        noise = em.get("noise_ratio", 0)
        iscore = im.get("score", 0)
        sep = im.get("separation", 0)
        stab = im.get("stability", 0)
        frag = im.get("fragmentation", 0)
        conf = run.get("confidence_level", "?")
        secs = run["seconds"]
        print(f"  [{tier:6s}] n={n:5d}  pred_k={pk:3d}  true={tc:2d}  "
              f"ARI={ari:.4f}  NMI={nmi:.4f}  V={vm:.4f}  "
              f"homo={homo:.4f}  comp={comp:.4f}  "
              f"cov={cov:.4f}  noise={noise:.4f}  "
              f"iscore={iscore:.4f}  sep={sep:.4f}  "
              f"stab={stab:.4f}  frag={frag:.4f}  "
              f"conf={conf}  secs={secs:.1f}")
    print()

# Now show per-cluster purity distributions for medium runs
print("\n========== PER-CLUSTER PURITY ANALYSIS (MEDIUM TIER) ==========\n")
for fname in sorted(os.listdir(results_dir)):
    if not fname.startswith("clusterer_") or not fname.endswith(".json"):
        continue
    with open(os.path.join(results_dir, fname)) as f:
        data = json.load(f)
    for run in data.get("runs", []):
        if run["tier_actual"] != "medium":
            continue
        emb = data.get("embedder_alias", "?")
        clusters = run.get("per_cluster", [])
        purities = [c["purity"] for c in clusters]
        sizes = [c["size"] for c in clusters]
        print(f"--- {emb} medium (pred_k={run['n_pred_clusters']}) ---")
        print(f"  Purity stats: min={min(purities):.4f}  max={max(purities):.4f}  "
              f"mean={sum(purities)/len(purities):.4f}  "
              f"median={sorted(purities)[len(purities)//2]:.4f}")
        print(f"  Size stats:   min={min(sizes)}  max={max(sizes)}  "
              f"mean={sum(sizes)/len(sizes):.0f}  "
              f"median={sorted(sizes)[len(sizes)//2]}")
        
        # Count garbage clusters (purity < 0.30)
        garbage = [(c["cluster_id"], c["size"], c["purity"], c["dominant_true_name"]) 
                   for c in clusters if c["purity"] < 0.30]
        if garbage:
            print(f"  GARBAGE CLUSTERS (purity < 0.30): {len(garbage)}")
            for gid, gsz, gp, gn in garbage:
                print(f"    cluster {gid}: size={gsz} purity={gp:.4f} dom={gn}")
        
        # Count over-split classes
        print(f"\n  TRUE CLASS RECOVERY:")
        recovery = run.get("true_class_recovery", [])
        for tc in recovery:
            frac = tc["best_cluster_fraction"]
            marker = " <-- FRAGMENTED" if frac < 0.50 else ""
            print(f"    {tc['true_name']:30s}  n={tc['n_docs']:4d}  best_frac={frac:.4f}  noise={tc['n_noise']}{marker}")
        print()
