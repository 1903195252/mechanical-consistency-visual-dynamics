"""Aggregate only complete three-seed groups, preserving each family and step."""
import json
import statistics
from .common import OUT,FAMILIES,SEEDS,STEPS,TEST_CASES,TEST_ASSEMBLIES_PER_FAMILY,CONDITIONS,write,sha


def run():
    rows=[];missing=[]
    for step in STEPS:
        reports=[]
        for seed in SEEDS:
            path=OUT/'runs'/('seed%d'%seed)/('step%04d'%step)/'scores.json'
            if not path.exists():missing.append(dict(seed=seed,step=step));continue
            report=json.loads(path.read_text())
            if report['cases']!=TEST_CASES or report['scoring_version']!='continuous-si-1':raise ValueError('Incomplete or incompatible score report')
            if report['seed']!=seed or report['step']!=step:raise ValueError('Seed/checkpoint mismatch')
            if report['protocol_sha256']!=sha(OUT/'protocol.json') or report['manifest_sha256']!=sha(OUT/'dataset_manifest.json'):
                raise ValueError('Dataset/protocol mismatch')
            reports.append(report)
        if len(reports)!=3:continue
        if len({r['measurement_profile_sha256'] for r in reports})!=1:raise ValueError('Measurement profiles differ across seeds')
        for family in FAMILIES:
            for branch in ('native','kinematic_intervention'):
                values=[r['families'][family]['branches'][branch]['continuous']['score_100'] for r in reports]
                rows.append(dict(step=step,family=family,branch=branch,seeds=list(SEEDS),
                    mean=statistics.mean(values),seed_std=statistics.stdev(values),per_seed=values,
                    independent_test_assemblies=TEST_ASSEMBLIES_PER_FAMILY,
                    trajectories_per_seed=TEST_ASSEMBLIES_PER_FAMILY*len(CONDITIONS),
                    variation_source='generation seed' if step==0 else 'training seed with paired generation seed',
                    finite_videos=[r['families'][family]['finite_videos'] for r in reports],
                    mechanics_coverage=[r['families'][family]['mechanics_coverage'][branch] for r in reports],
                    visual_per_seed=[r['families'][family]['visual'] for r in reports]))
    report=dict(complete=not missing,rows=rows,missing=missing,
        seed_variation='Step 0: generation-seed SD with identical base weights; step 50: independent training-seed SD with paired generation seeds',
        sampling_unit='assembly; conditions and diffusion repeats do not create additional identities')
    write(OUT/'comparison.json',report)
    lines=['# LingBot 50 + 50: paired 0/50-update audit','',
           '| Step | Family | Readout | Scomp mean ± seed SD |',
           '|---:|---|---|---:|']
    for row in rows:lines.append('| %d | %s | %s | %.3f ± %.3f |'%(row['step'],row['family'],row['branch'],row['mean'],row['seed_std']))
    lines+=['','Missing seed/checkpoint reports: %d. No incomplete three-seed group is averaged.'%len(missing),'']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    return report


if __name__=='__main__':run()
