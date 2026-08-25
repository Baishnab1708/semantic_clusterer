"""Extract all benchmark metrics from every result file for final release analysis."""
import json, os, sys

results_dir = os.path.join(os.path.dirname(__file__), 'results')

print("=" * 120)
print("FINAL RELEASE AUDIT — All Benchmark Results")
print("=" * 120)

for f in sorted(os.listdir(results_dir)):
    if not f.endswith('.json') or f.startswith('baseline'):
        continue
    path = os.path.join(results_dir, f)
    data = json.loads(open(path, encoding='utf-8').read())
    kind = data.get("kind", "?")
    emb = data.get("embedder_alias", "?")
    emb_name = data.get("embedder_name", "?")
    
    print(f"\n{'='*100}")
    print(f"FILE: {f}")
    print(f"KIND: {kind}  |  EMBEDDER: {emb_name}")
    print(f"ENV: {data.get('environment', {}).get('semantic-clusterer', '?')}")
    print(f"{'='*100}")
    
    for run in data.get('runs', []):
        # Handle both benchmark and production eval schemas
        em = run.get('external_metrics', {})
        im = run.get('intrinsic_metrics', {})
        
        # Production eval schema
        if 'ARI' in run:
            tier = run.get("tier", "?")
            phase = run.get("Phase", "?")
            thresh = run.get("Threshold", "?")
            ari = run.get("ARI", 0)
            nmi = run.get("NMI", 0)
            vm = run.get("V-Measure", 0)
            cov = run.get("Coverage", 0)
            noise = run.get("Noise Ratio", 0)
            pk = run.get("Pred K", 0)
            secs = run.get("Secs", 0)
            print(f"  {tier:8s} | {phase:15s} | Thresh={str(thresh):5s} | K={pk:3d} | ARI={ari:.4f} | NMI={nmi:.4f} | V={vm:.4f} | Cov={cov:.4f} | Noise={noise:.4f} | Time={secs:.2f}s")
        else:
            # Benchmark schema
            tier = run.get("tier_requested", "?")
            n = run.get("n_docs", 0)
            k = run.get("k", "auto")
            pk = run.get("n_pred_clusters", 0)
            noise_n = run.get("n_noise", 0)
            secs = run.get("seconds", 0)
            ari = em.get("ari", 0)
            nmi = em.get("nmi", 0)
            vm = em.get("v_measure", 0)
            hom = em.get("homogeneity", 0)
            comp = em.get("completeness", 0)
            cov = em.get("coverage", 1)
            nr = em.get("noise_ratio", 0)
            sc = im.get("score", 0)
            coh = im.get("cohesion", 0)
            sep = im.get("separation", 0)
            stab = im.get("stability", 0)
            sil = im.get("silhouette", "n/a")
            dbi = im.get("davies_bouldin", "n/a")
            
            print(f"  {tier:8s} | N={n:6d} | k={str(k):5s} | pred_K={pk:3d} | noise={noise_n:5d} | Time={secs:8.1f}s")
            print(f"           | ARI={ari:.4f} | NMI={nmi:.4f} | V={vm:.4f} | Hom={hom:.4f} | Comp={comp:.4f} | Cov={cov:.4f} | NoiseR={nr:.4f}")
            print(f"           | Score={sc:.4f} | Coh={coh:.4f} | Sep={sep:.4f} | Stab={stab:.4f} | Sil={sil} | DBI={dbi}")

print(f"\n{'='*120}")
print("END OF AUDIT")
print(f"{'='*120}")
