"""Post-hoc, model-independent kinetic-metric audit; no weights fitted or scored."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import lsq_linear
from experiments.mechanism_corpus.catalog import ROOT,OUTPUT,connect,write_json
from .prepare import OUT,point_states
from .physics_score import MODULES


def one(entry):
    folder=OUTPUT/entry['manifest_path'];folder=folder.parent;rows=[]
    for condition in ('low_speed','rated','limit_or_load'):
        params=json.loads((folder/'conditions'/condition/'physical_parameters.json').read_text())
        c=params['config'];module=MODULES[c['category']];features=[];targets=[]
        with np.load(folder/'conditions'/condition/'physics.npz') as data:
            positions=data['q'][240::18]  # Existing reference poses; no new simulation or video.
        for q in positions:
            J=module.closure_jacobian(q,c);N=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
            M,_,_,_,_=module.mass_bias_energy(q,np.zeros(3),c,params['bodies'])
            _,vel,_=point_states(q[None],N[None],np.zeros((1,3)),c)
            features.append(np.sum(vel[0]**2,axis=1));targets.append(N@M@N)
        A=np.asarray(features);y=np.asarray(targets)
        result=lsq_linear(A/y[:,None],np.ones(len(y)),bounds=(0,np.inf),tol=1e-12,max_iter=1000)
        relative=(A@result.x-y)/y
        rows.append({'assembly_id':entry['assembly_id'],'category':entry['category'],'condition':condition,
                     'poses':len(y),'feature_rank':int(np.linalg.matrix_rank(A)),
                     'best_positive_mass_metric_relative_rms':float(np.sqrt(np.mean(relative**2))),
                     'best_positive_mass_metric_relative_max':float(abs(relative).max()),
                     'oracle_fit_effective_point_masses_kg':result.x.tolist(),'features':A.tolist(),'target_metric_kg_m2':y.tolist()})
    return rows


def main():
    db=connect();entries=[dict(r) for r in db.execute('SELECT * FROM assemblies ORDER BY assembly_id') if r['category'] in MODULES];db.close()
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=[row for batch in pool.map(one,entries) for row in batch]
    A=np.concatenate([np.asarray(row['features'])/np.asarray(row['target_metric_kg_m2'])[:,None] for row in rows])
    fit=lsq_linear(A,np.ones(len(A)),bounds=(0,np.inf),tol=1e-12,max_iter=1000)
    shared=A@fit.x-1
    summaries={}
    for code in MODULES:
        values=[r['best_positive_mass_metric_relative_rms'] for r in rows if r['category']==code]
        summaries[code]={'conditions':len(values),'median_per_case_relative_rms':float(np.median(values)),
                         'maximum_per_case_relative_rms':float(np.max(values))}
    result={'model_result':False,'post_hoc_diagnostic':True,'assemblies':len(entries),'conditions':len(rows),
            'scope':'Fit the three physical B/C/midpoint masses to the true reduced kinetic metric, per case and shared across the corpus; existing physical reference poses only.',
            'interpretation':'These are evaluator oracle fits, not neural SI mass estimates. The learned potential/input matrix can compensate some inertial mismatch, so this is not a lower bound on rollout error and does not isolate optimization failure.',
            'per_family':summaries,'shared_mass_metric_relative_rms':float(np.sqrt(np.mean(shared**2))),
            'shared_oracle_effective_point_masses_kg':fit.x.tolist(),'rows':rows}
    write_json(OUT/'kinetic_capacity_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
