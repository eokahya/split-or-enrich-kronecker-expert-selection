"""Coordinator audit; imports no study code and never changes scientific outputs."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

root, run = map(Path, sys.argv[1:3])
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
header, summary = read(run/'run_header.json'), read(run/'summary.json')
frozen = header['freeze_manifest']
assert all(digest(root/p) == h for p,h in frozen['files'].items())
assert all(digest(run/p) == h for p,h in read(run/'output_manifest.json').items())
assert digest(run/'selections.json') == summary['selections_sha256']
assert header['mode'] == 'scientific' and header['root'] == 2026103101
assert header['env_ids'] == list(range(24)) and header['runtime']['torch_threads'] == 1
records = read(run/'records.json')
selections = read(run/'selections.json')
rows = list(csv.DictReader((run/'candidate_seed_losses.csv').open()))
assert len(rows) == 432 and len(records['test']) == 24 and len(selections) == 24
assert summary['complete'] and summary['valid'] and summary['unit_tests']['ok']
assert summary['total_runtime_s'] <= 3600
checks, views = {}, {}
max_k, max_loss, max_z = 0., 0., 0.
assignments = [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1),
               (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1), (0,1,2,3)]
for rec in records['test']:
    eid, Z = rec['env_id'], rec['Z']
    arr = np.load(run/'envs'/f'env{eid:02d}_train_eval.npz')
    t, sigma, noise = arr['T'], arr['Sigma'], arr['noise_var']
    expected_z = .125 * sum(np.trace(ti @ si @ ti.T) + 64*ni for ti,si,ni in zip(t,sigma,noise))
    max_z = max(max_z, abs(Z-expected_z)/Z)
    for c in range(9):
        A,B,K = arr[f'A_{c}'],arr[f'B_{c}'],arr[f'K_{c}']
        for seed in range(2):
            rebuilt = np.stack([sum(np.kron(a,b) for a,b in zip(aa,bb)) for aa,bb in zip(A[seed],B[seed])])
            max_k = max(max_k, float(np.max(np.abs(rebuilt-K[seed]))))
            loss = 0.
            for task,expert in enumerate(assignments[c]):
                k = rebuilt[expert]
                loss += .125*(np.trace(k @ arr['test_Sxx'][task] @ k.T)
                              -2*np.sum(k*arr['test_Syx'][task])+arr['test_syy'][task])
            max_loss = max(max_loss,abs(loss-rec['cand_loss'][seed][c])/Z)
    sel = selections[str(eid)]
    assert all(v == sel[k]['selected'] for k,v in rec['selected'].items())
    assert sel['early_ceiling']['status'] == 'MEASURED'
    assert sel['early_ceiling']['selection_elapsed_s'] <= 1.0
    assert sel['early_ceiling']['wall_s'] >= sel['early_ceiling']['selection_elapsed_s']
assert max_k < 1e-10 and max_loss < 1e-9 and max_z < 1e-12
for view in ['mean','seed0','seed1','min','population_risk_mean']:
    rs = records['population'] if view == 'population_risk_mean' else records['test']
    L = np.array([r['cand_loss'] for r in rs])
    losses = L[:,0] if view=='seed0' else L[:,1] if view=='seed1' else L.min(1) if view=='min' else L.mean(1)
    Z = np.array([r['Z'] for r in rs])
    regrets = (losses-losses.min(1)[:,None])/Z[:,None]
    idx=np.arange(24); an=np.array([r['cov_kind']=='aniso' for r in rs])
    naive = regrets[idx,[r['selected']['naive'] for r in rs]]
    means={f'fixed_{c}':float(regrets[an,c].mean()) for c in range(9)}
    for m in ['covariance','early_ceiling']:
        means[m]=float(regrets[idx,[r['selected'][m] for r in rs]][an].mean())
    count=int((naive<=.02).sum()); nm=float(naive[an].mean()); best=min(means.values())
    label='FAIL_PREDICTION' if count<20 else 'NO_ADDED_UTILITY' if nm>best-.01 else 'PASS'
    g=summary['verdict']['gates_by_aggregation'][view]
    assert count==g['gate2_envs_within_threshold'] and abs(nm-g['primary_aniso_mean_regret'])<1e-12
    assert abs(best-g['gate3_strongest_mean'])<1e-12
    assert label==summary['verdict']['labels_by_aggregation'][view]
    views[view]={'within_02':count,'naive_aniso':nm,'best_baseline':best,'label':label}
result={'status':'PASS','frozen_source_hashes_verified':len(frozen['files']),
        'output_manifest_verified':True,'selection_hash_verified':True,'rows':len(rows),
        'max_rebuilt_K_abs_error':max_k,'max_recomputed_test_loss_over_Z_error':max_loss,
        'max_normalizer_relative_error':max_z,'independent_gate_recalculation':views}
print(json.dumps(result,indent=2))
