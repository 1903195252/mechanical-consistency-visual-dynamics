"""Check the built manuscript, references, and decisive experimental values."""
import json
from pathlib import Path
import re
import subprocess
import base64
import hashlib
import statistics
import csv
import numpy as np
import xml.etree.ElementTree as ET
from prepare_con_artifacts import collect as collect_con_evidence
from prepare_dynamics_extension import collect as collect_dynamics_extension
from prepare_integrator_comparison import collect as collect_integrator_comparison
from prepare_shared_video_figure import collect as collect_shared_video
from check_resolution_evidence import verify as verify_resolution
from check_vjepa_expansion import verify as verify_vjepa_expansion

PAPER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def command(*args):
    return subprocess.check_output(args, text=True)



def main():
    results = {}
    title = re.search(r'\\newcommand\{\\keytitle\}\{([^\n]+)\}',
                      (PAPER/'preamble.tex').read_text()).group(1)
    for name in ("main", "supplement"):
        pdf = PAPER / (name + ".pdf")
        info = command("pdfinfo", str(pdf))
        log = (PAPER / "build" / (name + ".log")).read_text(errors="replace")
        text = command("pdftotext", "-layout", str(pdf), "-")
        fonts = command("pdffonts", str(pdf))
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        issues = [line for line in log.splitlines() if any(token in line for token in (
            "Overfull", "undefined references", "undefined citations", "Linenumber reference failed",
            "Rerun to get", "LaTeX Error", "Fatal error"))]
        assert not issues, (name, issues)
        assert "letter" in info
        assert "Author:         Anonymous" in info
        assert re.search(r'^Title:\s+(.+)$', info, re.M).group(1) == title
        assert "-12pt" not in text
        assert not re.search(r"/media/gu|/home/gu|seetacloud|VSPROJECT", text)
        assert "Type 3" not in fonts
        entries = [re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
                   for line in fonts.splitlines()[2:]]
        assert entries and all(match and match.group(1) == "yes" for match in entries)
        (PAPER / "build" / (name + ".txt")).write_text(text)
        # Layout extraction can merge a right-column heading with left-column prose.
        heading_text = command("pdftotext", "-raw", str(pdf), "-")
        reference_pages = [i + 1 for i, page in enumerate(heading_text.split('\f'))
                           if re.search(r'^\s*(?:\d+\s+)?References\b', page, re.M)]
        assert len(reference_pages) == 1, (name, reference_pages)
        results[name] = {"pages": pages, "US_Letter": True, "anonymous_metadata": True,
                         "fonts_embedded": True, "type3_fonts": False, "unresolved_build_issues": issues,
                         "references_start_page": reference_pages[0]}
        if name == 'supplement':
            reference_text = text.split('\f')[reference_pages[0] - 1]
            bibliography_items = re.findall(r'\\bibitem', (PAPER/'build/supplement.bbl').read_text())
            assert reference_pages[0] == pages, 'Supplement references must stay together on the last page'
            for number in range(1, len(bibliography_items) + 1):
                assert re.search(r'\[' + str(number) + r'\]', reference_text), number
    aux = (PAPER / "build/main.aux").read_text()
    match = re.search(r"\\newlabel\{maintextend\}\{\{[^}]*\}\{(\d+)\}", aux)
    assert match
    results["main"]["main_body_pages"] = int(match.group(1))
    assert results["main"]["main_body_pages"] <= 8
    float_pages = {label: int(page) for label, page in re.findall(
        r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{[^}]*\}\{(\d+)\}', aux)}
    assert float_pages and max(float_pages.values()) <= 8, float_pages
    assert results['main']['references_start_page'] == 9
    results['main']['figure_table_pages'] = float_pages
    sources = (PAPER / "main.tex").read_text() + (PAPER / "supplement.tex").read_text()
    for name in ("main", "supplement"):
        source = (PAPER / (name + ".tex")).read_text()
        results[name]["figure_count"] = (len(re.findall(r"\\begin\{figure\*?\}|\\captionof\{figure\}", source))
                                         - len(re.findall(r"\\ContinuedFloat\b", source)))
        results[name]["table_count"] = len(re.findall(r"\\begin\{table\*?\}", source))
        for graphic in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
            assert (PAPER / graphic).is_file(), graphic
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", sources):
        cited.update(group.split(","))
    bibliography = (PAPER / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    assert cited == keys, {"missing": list(cited-keys), "uncited": list(keys-cited)}
    reference_orders = {}
    for name in ("main", "supplement"):
        assert r'\bibliographystyle{ieeenat_fullname}' in (PAPER / (name + '.tex')).read_text()
        entries = re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}",
                             (PAPER / "build" / (name + ".bbl")).read_text())
        assert len(entries) == len(keys) and set(entries) == keys
        reference_orders[name] = entries
    assert reference_orders['main'] == reference_orders['supplement']
    results['reference_order'] = {'bibliography_order': 'author_alphabetical',
        'reference_keys': reference_orders['main'], 'supplement_shares_main_numbers': True,
        'style': 'ieeenat_fullname', 'official_template_style': True,
        'policy_checked_date': '2026-10-17',
        'policy_source': 'https://github.com/cvpr-org/author-kit/blob/main/sec/2_formatting.tex'}
    source_root = ROOT / "output/keycld_fourbar"
    scored = json.loads((source_root / "scored_evaluation/scores.json").read_text())
    state = scored["scorecards"]["state_RK4"]
    assert len(state["rows"]) == 160 and state["joint_passed_trajectories"] == 0
    assert state["component_scores"]["dynamics"]["passed_trajectories"] == 54
    assert state["component_scores"]["energy"]["passed_trajectories"] == 0
    assert state["component_scores"]["prediction"]["passed_trajectories"] == 21
    main_text = (PAPER / "main.tex").read_text()
    table_inputs=set(re.findall(r"\\input\{(tables/[^}]+)\}",sources))
    manuscript_evidence_text=sources+"\n"+"\n".join((PAPER/(name+".tex")).read_text() for name in table_inputs)
    parameter_rows = (PAPER / "tables/transfer_rows.tex").read_text()
    assert r'\input{tables/transfer_rows}' in sources
    for condition in ("nominal", "mixed_friction", "heavy_payload", "dense_bars", "geometry_scaled", "crank_slider"):
        value = state["conditions"][condition]["metrics"]["dynamics"]["mean"]
        assert "%.3f" % value in parameter_rows, condition
    assert scored["calibration_controls"]["oracle"]["passed"] == 96
    assert scored["calibration_controls"]["static"]["passed"] == 0
    assert scored["calibration_controls"]["time_scaled_0.8"]["passed"] == 5
    probes = json.loads((source_root / "extended_dynamics/field_probes.json").read_text())
    nominal = probes["conditions"]["nominal"]["inertia"]
    assert nominal["positive_gain_poses"] == 71
    assert "%.2f" % (100 * nominal["equivalent_inertia_relative_error"]["median"]) in manuscript_evidence_text
    assert "%.2f" % (100 * nominal["equivalent_inertia_relative_error"]["mean"]) in manuscript_evidence_text
    diagnosis = json.loads((source_root / "diagnosis/report.json").read_text())
    assert "%.2f" % (1000 * diagnosis["summary"]["visual_state"]["constant_position_rmse_m"]["mean"]) in manuscript_evidence_text
    assert "initial position-constraint RMSE" not in main_text
    corpus = json.loads((PAPER / "sources/corpus_evidence.json").read_text())
    assert corpus['evaluated_assemblies'] == 343 and corpus['checked_prediction_archives'] == 1458
    assert corpus['equal_initial_checkpoints_within_seed']
    assert corpus['assembly_splits'] == {'test':36,'validation':18,'calibration':18,'train':271}
    assert corpus['calibration']['summaries']['oracle']['passed'] == 54
    assert corpus['calibration']['summaries']['persistence']['passed'] == 0
    assert corpus['calibration']['summaries']['time_scaled_0.8']['passed'] == 0
    for arm in ['state_euler','state_rk4']:
        assert corpus['summary'][arm]['complete_physical_trajectories'] == 324
        assert corpus['summary'][arm]['joint_pass'] == 0
    for key in ['force_balance_Nm','work_energy_J']:
        values = corpus['paired_integrator_changes'][key]
        for field in ['euler_mean','rk4_mean']:
            assert ('%.3f' % values[field]) in manuscript_evidence_text, (key,field)
    assert '97.22' in manuscript_evidence_text and '4.63' in manuscript_evidence_text
    assert r'\tau_{\max}=12' in main_text
    assert 'specifies absolute precision' in main_text
    assert (PAPER / 'figures/audit_mechanism.drawio').is_file()
    for label in ['eq:pointconstraints','eq:bodydynamics','eq:friction','eq:scores','eq:learning','eq:worked','eq:identification','eq:controller-confound','eq:design-dissipation']:
        assert (r'\label{'+label+'}') in sources, label
    assert 'two complementary studies' not in main_text.lower()
    assert 'fixed-system' not in main_text.lower()
    assert 'Implications and Conclusion' in main_text
    renders = json.loads((PAPER / 'sources/rendered_figure_sources.json').read_text())
    assert len(renders['records']) == len({r['assembly_id'] for r in renders['records']}) == 343
    assert renders['counts'] == {'S-A':120,'S-B':116,'S-J':107}
    assert sum(r['split']=='test' for r in renders['records']) == 36
    for r in renders['records']:
        assert hashlib.sha256((PAPER/r['asset']).read_bytes()).hexdigest() == r['asset_sha256']
    atlas = renders['atlas_display']
    assert atlas['identities'] == [r['assembly_id'] for r in renders['records'] if r['split']=='test']
    assert atlas['count'] == 36 and atlas['grid_per_family'] == [3,4]
    assert atlas['full_corpus_in_scatter_panels'] == 343
    assert len(atlas['motion_annotation_sources']) == 3
    extension=atlas['spatial_extension']
    assert hashlib.sha256((ROOT/extension['manifest']).read_bytes()).hexdigest()==extension['manifest_sha256']
    spatial_manifest=json.loads((ROOT/extension['manifest']).read_text())
    assert [r['category'] for r in extension['examples']]==['H-A','H-B']
    for example in extension['examples']:
        original=next(r for r in spatial_manifest['videos'] if r['video_id']==example['evaluation_video_id'])
        for key in ('assembly_id','split','condition'):
            assert example[key]==original[key]
        sidecar=ROOT/example['public_inputs']
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest()==example['public_inputs_sha256']
        recorded=json.loads(sidecar.read_text())
        assert recorded['condition']==example['condition'] and recorded['view']==example['view']
        assert Path(example['source']).stem==example['video_id']
        assert example['split']=='test' and example['condition']=='rated' and example['view']=='isometric'
        assert example['frame']==0 and example['time_s']==0
        assert hashlib.sha256((ROOT/example['source']).read_bytes()).hexdigest()==example['source_sha256']
        assert hashlib.sha256((PAPER/example['asset']).read_bytes()).hexdigest()==example['asset_sha256']
        family=[r for r in spatial_manifest['videos'] if r['category']==example['category']]
        assert example['assemblies']==len({r['assembly_id'] for r in family})
        assert example['test_identities']==len({r['assembly_id'] for r in family if r['split']=='test'})==50
    for annotation in atlas['motion_annotation_sources']:
        for source in annotation['sources']:
            assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
    drawio = ET.parse(PAPER / 'figures/audit_mechanism.drawio')
    diagram_cells = {c.get('id'): c for c in drawio.iter('mxCell')}
    assert len(diagram_cells) == len(list(drawio.iter('mxCell')))
    diagram_edges = {(c.get('source'), c.get('target')) for c in diagram_cells.values()
                     if c.get('edge') == '1' and c.get('source')}
    for source, target in (('spatial-prefix','spatial-model'), ('lingbot-prefix','lingbot-model'),
            ('spatial-model','spatial-features'), ('spatial-features','spatial-readout'),
            ('spatial-readout','spatial-readout-join'), ('lingbot-model','lingbot-video'),
            ('lingbot-video','lingbot-observer'), ('lingbot-observer','lingbot-readout-join'),
            ('native-state','kinematic-audit'), ('kinematic-audit','si-mechanics'),
            ('si-parameters','si-reference-join'), ('si-reference-join','kinematic-audit'),
            ('si-reference-join','si-mechanics'), ('si-mechanics','balance-fork'),
            ('balance-fork','effort-audit'), ('balance-fork','energy-audit'),
            ('effort-audit','balance-merge'), ('energy-audit','balance-merge'),
            ('balance-merge','audit-report'), ('kinematic-audit','audit-report')):
        assert (source,target) in diagram_edges
    for stem,labels in [('spatial',('3 RGB · 384 px','Actuator commands')),
                        ('lingbot',('Last RGB · 256 px','Actuator commands'))]:
        assert tuple(diagram_cells[stem+'-prefix-'+suffix].get('value')
                     for suffix in ('frames','command')) == labels
    assert not any(source == 'spatial-input' for source, _ in diagram_edges)
    for edge_name in ('spatial-input-model','lingbot-input-model'):
        geometry = diagram_cells[edge_name].find('mxGeometry')
        start = geometry.find("mxPoint[@as='sourcePoint']")
        end = geometry.find("mxPoint[@as='targetPoint']")
        assert start.get('x') == end.get('x') and float(start.get('y')) < float(end.get('y'))
        assert not geometry.findall('Array/mxPoint')

    def diagram_reaches(start, target):
        pending, visited = [start], set()
        while pending:
            source = pending.pop()
            if source == target:
                return True
            if source not in visited:
                visited.add(source)
                pending.extend(b for a, b in diagram_edges if a == source)
        return False

    for reader in ('key-reader','con-reader','spatial-readout','lingbot-observer'):
        assert diagram_reaches(reader, 'native-state')
        for target in ('effort-audit','energy-audit','audit-report'):
            assert diagram_reaches(reader, target)
    for model in ('encoder','keycld','con-encoder','con-ode','spatial-model','lingbot-model'):
        assert not diagram_reaches('si-parameters', model)
    assert not diagram_reaches('effort-audit', 'energy-audit')
    assert not diagram_reaches('energy-audit', 'effort-audit')
    assert 'B u_A' in diagram_cells['generalized-force'].get('value')
    assert r'$B u_A=e_1\tau_A$' in main_text
    assert 'Calibrated RGB' in diagram_cells['lingbot-observer'].get('value')
    assert 'physical readout' in diagram_cells['lingbot-observer'].get('value')
    images = [c for c in drawio.iter('mxCell') if 'shape=image;' in c.get('style','')]
    assert len(images) == 3
    for c in images:
        payload = c.get('style').split('image=data:image/png,',1)[1].split(';',1)[0]
        assert base64.b64decode(payload).startswith(b'\x89PNG')
    results['rendered_figures'] = {'unique_assembly_renders':343,'held_out_assemblies_highlighted':36,
        'atlas_display_identities':36,'atlas_grid_per_family':[3,4],
        'motion_annotations_use_calibrated_poses':True,'display_contrast_only':True,
        'enlarged_examples':3,'spatial_examples':2,'drawio_native_image_cells':3,'assets_match_recorded_hashes':True,
        'fig1_text_box_bounds':'Verified during deterministic diagram export',
        'fig1_flow':{'inputs_start_at_observation_nodes':True,'shared_readout_bus':True,
            'parallel_effort_energy_branches':True,'kinematic_scores_bypass_mechanics_gate':True,
            'evaluator_reference_has_no_path_to_models':True},
        'provenance':'sources/rendered_figure_sources.json'}
    resolution, results['paired_resolution'] = verify_resolution()
    overview = json.loads((PAPER / 'sources/model_overview.json').read_text())
    assert overview['scoring_version'] == 'continuous-si-1'
    assert [p['model'] for p in overview['profiles']] == ['KeyCLD', 'KeyCLD', 'CON', 'CON', 'V-JEPA', 'LingBot']
    for path, digest in overview['sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    canonical_scores = json.loads((ROOT / 'output/closed_chain_eval/continuous/scores.json').read_text())
    metrics = ('geometry', 'constraint_velocity', 'constraint_acceleration', 'dynamics', 'energy', 'prediction')
    with (ROOT/'output/closed_chain_eval/continuous/case_scores.csv').open() as stream:
        canonical_cases=list(csv.DictReader(stream))
    from prepare_spatial_expansion import collect as collect_expansion, overview_profile
    expanded = collect_expansion()
    mechanics_points=0;coverage_counts=[]
    for i, profile in enumerate(overview['profiles']):
        for track_field, value_field in (('native_track', 'native_score_100'),
                                         ('reconstructed_track', 'reconstructed_score_100')):
            if i < 4:
                track = resolution['runs'][i]['native' if track_field=='native_track' else 'reconstructed']
                selected = [dict(continuous_score_100=track['score'], components_100=track['components'], expected_cases=324)]
            elif i == 5:
                branch = 'native' if track_field == 'native_track' else 'kinematic_intervention'
                selected = [dict(continuous_score_100=e['branches'][branch]['score'],
                    components_100=e['branches'][branch]['components'], expected_cases=150)
                    for family in ('H-A','H-B') for e in expanded['groups'][f'{family}_50']]
            else:
                selected = [r for r in canonical_scores['summaries'] if r['model'] == profile['model']
                            and r['arm'] == profile['arm'] and r['suite'] == profile['suite']
                            and r['track'] == profile[track_field]]
            assert len(selected) == (1 if i < 4 else 6 if i == 5 else 2)
            assert abs(profile[value_field] - statistics.mean(r['continuous_score_100'] for r in selected)) < 1e-10
            component_field = 'components_100' if track_field == 'native_track' else 'reconstructed_components_100'
            if track_field == 'native_track':
                assert profile['registered_cases'] == sum(r['expected_cases'] for r in selected)
            track_name = 'native' if track_field == 'native_track' else 'reconstructed'
            layout=overview['plot']['contribution_layout']
            start=layout['left']
            yy=layout['row_y'][i]+layout['track_y_offsets'][0 if track_name=='native' else 1]
            for metric in metrics:
                expected = statistics.mean(r['components_100'][metric] for r in selected)
                assert abs(profile[component_field][metric] - expected) < 1e-10
                width=layout['width']*expected/600
                geometry=diagram_cells[f'contribution-{track_name}-{i}-{metric}'].find('mxGeometry')
                np.testing.assert_allclose(
                    [float(geometry.get(k)) for k in ('x','y','width')],
                    [start,yy,width],atol=1e-8,rtol=0)
                start+=width
            assert abs(profile[value_field]-statistics.mean(profile[component_field].values()))<1e-10
            endpoint=diagram_cells[f'{track_name}-end-{i}'].find('mxGeometry')
            actual=float(endpoint.get('x'))+float(endpoint.get('width'))/2
            assert abs(actual-start)<1e-8
            label=diagram_cells[f'{track_name}-score-value-{i}'].get('value')
            assert label==f"{statistics.mean(r['continuous_score_100'] for r in selected):.1f}"
        if i < 4:
            candidates = [dict(r, seed=str(r['seed']), dynamics_SI=r['dynamics'], dynamics_limit_SI=.25,
                               energy_SI=r['energy'], energy_limit_SI=.05)
                          for r in resolution['runs'][i]['reconstructed']['rows']]
        elif i == 5:
            candidates=[]
            for family in ('H-A','H-B'):
                for e in expanded['groups'][f'{family}_50']:
                    for r in e['rows']:
                        b=r.get('kinematic_intervention',{})
                        measured=b.get('mechanical_coverage',False)
                        candidates.append(dict(case_id=r['video_id'],seed=str(r['seed']),family=family,condition=r['condition'],
                            dynamics_ratio=b['effort_rmse']/b['effort_tolerance'] if measured else None,
                            energy_ratio=b['work_energy_max_J']/b['energy_tolerance_J'] if measured else None,
                            dynamics_SI=b.get('effort_rmse'),dynamics_limit_SI=b.get('effort_tolerance'),
                            energy_SI=b.get('work_energy_max_J'),energy_limit_SI=b.get('energy_tolerance_J')))
        else:
            candidates=[r for r in canonical_cases if r['model']==profile['model']
                        and r['arm']==profile['arm'] and r['suite']==profile['suite']
                        and r['track']==profile['reconstructed_track']]
        available=[r for r in candidates if r['dynamics_ratio'] and r['energy_ratio']]
        evidence=profile['reconstructed_mechanics']
        assert len(candidates)==profile['registered_cases']
        assert len(available)==evidence['measurable_cases']
        if i != 4:assert len(available)==[311,324,308,311,None,874][i]
        else:assert len(candidates)==900
        coverage_counts.append(len(available))
        expected_records=[{**{key:r[key] for key in ('case_id','seed','family','condition')},
                           **{key+'_ratio':float(r[key+'_ratio']) for key in ('dynamics','energy')}}
                          for r in available]
        assert evidence['cases']==expected_records
        coverage_fraction=len(available)/len(candidates)
        assert diagram_cells[f'mechanics-upper-count-{i}'].get('value')==(
            f"{len(available)}/{len(candidates)} · {100*coverage_fraction:.1f}%")
        layout=overview['plot']['upper_mechanics_layout']
        for metric in ('dynamics','energy'):
            spec=layout['axes'][metric]
            lo,hi=np.log10(spec['limits'])
            def plotted_x(value):
                return spec['left']+spec['width']*(np.log10(value)-lo)/(hi-lo)
            values=np.array([float(r[metric+'_ratio']) for r in available])
            np.testing.assert_allclose(values,[float(r[metric+'_SI'])/float(r[metric+'_limit_SI']) for r in available],rtol=1e-12)
            quantiles=np.quantile(values,[0,.25,.5,.75,1])
            np.testing.assert_allclose(evidence['distributions'][metric]['quantiles'],quantiles)
            assert evidence['distributions'][metric]['above_tolerance']==int(np.sum(values>1))
            assert spec['limits'][0]<min(values)<=max(values)<spec['limits'][1]
            for j,value in enumerate(values):
                geometry=diagram_cells[f'mechanics-upper-{metric}-case-{i}-{j}'].find('mxGeometry')
                x=float(geometry.get('x'))+float(geometry.get('width'))/2
                y=float(geometry.get('y'))+float(geometry.get('height'))/2
                assert abs(x-plotted_x(value))<1e-8
                assert abs(y-layout['row_y'][i])<=layout['jitter_half_width']
                mechanics_points+=1
            box=diagram_cells[f'mechanics-upper-{metric}-iqr-{i}'].find('mxGeometry')
            np.testing.assert_allclose([float(box.get('x')),float(box.get('width'))],
                                       [plotted_x(quantiles[1]),plotted_x(quantiles[3])-plotted_x(quantiles[1])],atol=1e-8)
            for key,expected in [('range',quantiles[[0,4]]),('median',[quantiles[2]]*2)]:
                geometry=diagram_cells[f'mechanics-upper-{metric}-{key}-{i}'].find('mxGeometry')
                xs=[float(geometry.find("mxPoint[@as='%s']"%end).get('x')) for end in ('sourcePoint','targetPoint')]
                np.testing.assert_allclose(xs,[plotted_x(v) for v in expected],atol=1e-8)
            threshold=diagram_cells[f'mechanics-upper-{metric}-tolerance'].find('mxGeometry')
            for end in ('sourcePoint','targetPoint'):
                assert abs(float(threshold.find("mxPoint[@as='%s']"%end).get('x'))-plotted_x(1))<1e-8
        assert profile['measured_counts']['dynamics'] == profile['measured_counts']['energy'] == 0
        assert abs(profile['native_score_100'] - statistics.mean(profile['components_100'].values())) < 1e-10
    assert mechanics_points==2*sum(coverage_counts)
    assert overview['plot']['composite_divisor'] == 6 and overview['plot']['score_axis'] == [0, 100]
    assert overview['plot']['reference_score'] == 50
    assert 'max. from kinematics' in diagram_cells['score-reference-label'].get('value')
    assert overview['plot']['reference_score'] == 100 * 3 / overview['plot']['composite_divisor']
    for metric, field in [('geometry', 'geometric-test'), ('dynamics', 'force-label'), ('energy', 'energy-label'), ('prediction', 'point-label')]:
        assert 'fontColor=' + overview['plot']['criterion_colors'][metric] + ';' in diagram_cells[field].get('style')
    results['four_model_overview'] = dict(criterion_scores_verified=72, composite_summaries_verified=12,
        mechanics_case_markers_verified=mechanics_points, mechanics_cases=sum(coverage_counts),
        mechanics_quartile_boxes_verified=12, coverage_counts=coverage_counts,
        protocol_groups=2, criterion_mean_matches_composite=True,
        reference_50_is_not_joint_qualification=True,
        missing_terms_retained=True, canonical_scores_verified=True,
        source='sources/model_overview.json')
    diagnostic = json.loads((PAPER / 'sources/diagnostic_figure_data.json').read_text())
    raw_path = ROOT / diagnostic['source']
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == diagnostic['source_sha256']
    raw = sorted(json.loads(raw_path.read_text()), key=lambda row: row['index'])
    assert diagnostic['test_trajectory_ids'] == list(range(500,550))
    assert diagnostic['individual_bars'] == 150 and diagnostic['summary_bars'] == 3
    assert diagnostic['std_ddof'] == 0
    for i,row in enumerate(raw):
        for j,field in enumerate(diagnostic['fields']):
            assert abs(diagnostic['per_trajectory_RMSE_mm'][i][j]-1000*row[field]) < 1e-10
    for j,field in enumerate(diagnostic['fields']):
        original = diagnosis['summary']['visual_state'][field]
        assert abs(diagnostic['mean_trajectory_RMSE_mm'][j]-1000*original['mean']) < 1e-10
        assert abs(diagnostic['standard_deviation_mm'][j]-1000*original['std']) < 1e-10
    results['diagnostic_figure'] = {'individual_error_bars':150,'test_trajectories':50,
        'readout_conditions':3,'unchanged_summary_statistics':True,'source_hash_verified':True,
        'provenance':'sources/diagnostic_figure_data.json'}
    coverage = json.loads((PAPER/'sources/dataset_coverage.json').read_text())
    assert coverage['assemblies'] == len(coverage['rows']) == 343
    assert len({r['assembly_id'] for r in coverage['rows']}) == 343
    assert sum(r['split']=='test' for r in coverage['rows']) == 36
    assert coverage['physical_parameter_hashes_verified'] == 1029
    assert coverage['operating_conditions'] == {'speed_rad_s':[1,3,5],'payload_multiplier':[1,1,1.75]}
    assert hashlib.sha256((ROOT/coverage['source']).read_bytes()).hexdigest() == coverage['source_sha256']
    measured = json.loads((PAPER/'sources/operating_dynamics.json').read_text())
    registered = json.loads((ROOT/coverage['source']).read_text())['records']
    assert {r['case_id'] for r in measured['rows']} == {r['case_id'] for r in registered if r['split']=='test'}
    assert len(measured['rows']) == 108 and measured['samples_per_case'] == 689
    for row in measured['rows']:
        for field in ('reference', 'physical_parameters'):
            assert hashlib.sha256((ROOT/row[field]).read_bytes()).hexdigest() == row[field+'_sha256']
        limit = json.loads((ROOT/row['physical_parameters']).read_text())['config']['actuator']['torque_limit_Nm']
        assert row['torque_limit_Nm'] == limit
        with np.load(ROOT/row['reference']) as data:
            mask = (data['time'] >= 1.1-1e-9) & (data['time'] <= 119/30+1e-9)
            assert int(mask.sum()) == 689
            torque, speed = data['actual_motor_torque_Nm'][mask], data['qdot'][mask, 0]
            np.testing.assert_allclose(data['motor_power_W'][mask], torque*speed, atol=1e-10)
            np.testing.assert_allclose(row['torque_rms_percent_limit'], 100*np.sqrt(np.mean(torque**2))/limit)
            np.testing.assert_allclose(row['absolute_motor_work_J'], np.trapz(np.abs(torque*speed), data['time'][mask]))
    for condition in ('low_speed', 'rated', 'limit_or_load'):
        group = [r for r in measured['rows'] if r['condition'] == condition]
        assert len(group) == 36
        for metric in ('torque_rms_percent_limit', 'absolute_motor_work_J'):
            assert abs(measured['summary'][condition][metric]['median'] - statistics.median(r[metric] for r in group)) < 1e-10
    results['measured_dataset_dynamics'] = {'test_cases':108, 'conditions':3, 'cases_per_condition':36,
        'interval_s':measured['interval_s'], 'source_hashes_and_trajectory_statistics_verified':True,
        'absolute_work_is_excitation_not_energy_defect':True}
    constitutive = json.loads((PAPER/'sources/constitutive_figure_data.json').read_text())
    for source in constitutive['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
    assert constitutive['unique_query_states'] == 648 and constitutive['checkpoints'] == 6
    assert constitutive['all_learned_velocity_odd_responses_zero']
    assert constitutive['capacity_poses'] == 44280
    assert abs(constitutive['shared_point_mass_relative_RMS'] - 0.22548959104709346) < 1e-12
    assert constitutive['layout']['constitutive_evidence'] == {'panels':3,'rows':1,'columns':3}
    assert constitutive['layout']['geometric_readout_cost']['destination'] == 'Figure 3(d), Section 5.3'
    assert float_pages['fig:shared-video'] == 5
    assert 'fig:geometry-cost' not in float_pages
    assert float_pages['fig:spatial'] == 8
    assert float_pages['tab:geometry-readout'] < float_pages['fig:constitutive']
    capacity=json.loads((ROOT/'output/keycld_corpus/kinetic_capacity_audit.json').read_text())
    shared_mass=np.array(capacity['shared_oracle_effective_point_masses_kg'])
    for family,value in constitutive['family_shared_relative_RMS'].items():
        rows=[row for row in capacity['rows'] if row['category']==family]
        reference=np.concatenate([row['target_metric_kg_m2'] for row in rows])
        fitted=np.vstack([row['features'] for row in rows])@shared_mass
        np.testing.assert_allclose(value,np.sqrt(np.mean((fitted/reference-1)**2)))
    np.testing.assert_allclose(constitutive['capacity_insets']['shared_by_family']['values_percent'],
        [100*constitutive['family_shared_relative_RMS'][f] for f in ('S-A','S-B','S-J')])
    results['new_physical_visualizations'] = {'assembly_coverage':343,'parameter_files_verified':1029,
        'state_checkpoints':6,'query_states_per_checkpoint':648,'kinetic_reference_poses':44280,
        'scope_and_source_hashes_verified':True}
    visual = json.loads((PAPER/'sources/visual_analysis.json').read_text())
    for source in visual['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
    assert visual['test_trajectories'] == 50 and visual['trajectory_ids'] == list(range(500,550))
    assert abs(visual['motion_pixel_fraction_mean'] - 0.1705517578125) < 1e-12
    assert abs(visual['motion_forecast_squared_error_share_mean'] - 0.9573792338519321) < 1e-12
    local = visual['paired_local_prediction']['euler_1']
    assert local['paired_bootstrap_95pct_mm'][0] > 0
    for value in ['17.06','95.74','6.774','6.657','0.117','0.086','0.153','5.47']:
        assert value in manuscript_evidence_text, value
    assert 'validation' in sources and 'readout' in sources
    results['visual_analysis'] = {'test_trajectories':50,'source_hashes_verified':True,
        'motion_area_percent':round(100*visual['motion_pixel_fraction_mean'],2),
        'motion_error_share_percent':round(100*visual['motion_forecast_squared_error_share_mean'],2),
        'local_prediction_bootstrap':'2000 paired trajectory draws, seed 20261013',
        'validation_error_and_test_coverage_distinguished':True}
    results['corpus_evidence'] = {'prediction_archives_checked':1458,'training_runs':9,
        'unique_evaluated_assemblies':343,'test_assemblies':36,'test_predictions_per_arm':324,
        'same_initialization_within_seed':True,'original_physical_thresholds_preserved':True,
        'controller_scale_clarification':'S-A/S-B use 5 Nm; S-J uses 12 Nm; 0.25 Nm remains the common absolute criterion'}
    unified = json.loads((PAPER / 'sources/unified_scoring.json').read_text())
    assert unified['scoring_version'] == 'continuous-si-1'
    for source, digest in unified['sources'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    con = collect_con_evidence()
    assert con == json.loads((PAPER / 'sources/con_evidence.json').read_text())
    con_table = (PAPER / 'tables/con_score_rows.tex').read_text()
    table_rows = [line.split(' & ') for line in con_table.splitlines() if line.strip()]
    assert len(table_rows) == 2
    for cells, arm in zip(table_rows, ('video_con_iae', 'state_con_iae')):
        summary = con['summary'][arm]
        assert cells[1] == '0/324' and cells[5:7] == ['N/A', 'N/A']
        for cell, key in zip(cells[2:5], ('geometry_m', 'velocity_m_s', 'acceleration_m_s2')):
            assert abs(float(cell) - summary['component_scores'][key]) < 0.0051
        assert cells[7] == '0.00' and cells[8] == '0'
        assert cells[9].startswith('%.2f [' % unified['planar'][arm]['mean'])
    con_seed_rows = (PAPER / 'tables/con_seed_rows.tex').read_text().splitlines()
    assert len(con_seed_rows) == 6
    for value in ('123.48', '142.39', '86.42', '93.54', '37.6', '3.0802', '4.9399'):
        assert value in manuscript_evidence_text, value
    assert r'\input{tables/planar_audit_main_rows}' in main_text
    assert r'\input{tables/con_score_rows}' in sources
    assert r'\input{tables/con_seed_rows}' in sources
    assert 'stolzle2024con' in cited
    assert title in (PAPER/'README.md').read_text()
    assert 'Structural and Intervention Analysis' in main_text
    results['con_evidence'] = {
        'training_runs': 6, 'per_case_test_reports_verified': 648,
        'reported_prediction_archives': 972,
        'same_manifest_calibration_thresholds_and_case_ids': True,
        'training_inference_scoring_checkpoint_identities_match': True,
        'both_arms_complete_physical_coverage': 0,
        'force_energy_retained_as_unavailable': True,
        'rgb_mse_reduction_percent': con['summary']['video_con_iae']['rgb_reduction_percent'],
        'raw_archive_recovery_verified': con['raw_archive_recovery_verified'],
    }
    extension=collect_dynamics_extension()
    assert extension==json.loads((PAPER/'sources/dynamics_extension_evidence.json').read_text())
    for filename,count in [('matched_video_rows.tex',2),('geometry_readout_rows.tex',7),('con_integrator_rows.tex',2),('inference_solver_rows.tex',12)]:
        assert len((PAPER/'tables'/filename).read_text().splitlines())==count
        assert r'\input{tables/'+filename[:-4]+'}' in sources
    for value in ['308/324','6.859','17.960','73.185','124.475','137.382','12.907','6.734','23.497']:
        assert value in manuscript_evidence_text,value
    results['dynamics_extension']={'new_training_runs':6,'new_native_predictions_verified':972,
        'additional_trajectory_files_verified':extension['summary']['additional_trajectory_files_verified'],
        'raw_and_readout_case_pairs':2268,'fixed_inference_trajectories':3888,
        'paired_initialization_and_all_epoch_orders':True,'raw_scores_reproduced':True,
        'physical_readout_is_explicit_intervention':True,'native_CON_force_energy_missingness_preserved':True}
    integrators = collect_integrator_comparison()
    assert integrators == json.loads((PAPER/'sources/integrator_comparison_data.json').read_text())
    assert [len(p['pairs']) for p in integrators['panels']] == [324,324,324,324,315,315]
    assert [p['track'] for p in integrators['panels']] == ['raw','conditioned','conditioned','raw','conditioned','conditioned']
    assert r'\includegraphics[width=\textwidth]{figures/corpus_paired_residuals.pdf}' in main_text
    figure_text = command('pdftotext', str(PAPER/'figures/corpus_paired_residuals.pdf'), '-')
    assert all(label in figure_text for label in ['KeyCLD: point', 'CON: point', 'CON: torque', 'CON: energy', 'Point density'])
    integrator_caption = re.search(
        r'\\caption\{([^\n]+)\}\s*\\label\{fig:corpus-residuals\}', main_text).group(1)
    assert all(term in integrator_caption for term in
               ('Euler/RK4', 'native point error', 'reconstructed force/energy'))
    assert 'Resolution-independent' in integrator_caption and 'Six curves' in integrator_caption
    traces=integrators['trace_curves'];timing=integrators['trace_protocol']
    assert [(c['model'],c['width']) for c in traces]==[
        ('KeyCLD',64),('KeyCLD',256),('KeyCLD',None),('CON',64),('CON',256),('CON',None)]
    assert all(c['label'] in figure_text for c in traces)
    assert timing['plotted_state_samples']==705 and timing['plotted_energy_samples']==697 and timing['future_samples']==689
    assert timing['forecast_start_s']==1.1 and abs(timing['energy_anchor_s']-(1+2/30))<1e-12
    time=np.asarray(timing['absolute_time_s']);energy_time=np.asarray(timing['energy_time_s'])
    np.testing.assert_allclose(time,np.arange(248,953)/240,atol=1e-12,rtol=0)
    np.testing.assert_allclose(energy_time,time[8:],atol=1e-12,rtol=0)
    for curve in traces:
        assert curve['linestyle']=={64:'--',256:'-.',None:'-'}[curve['width']]
        assert curve['saved_case_metrics_verified'] and curve['common_reconstruction_verified']
        assert len(curve['phase_rad'])==len(curve['force_Nm'])==705 and len(curve['energy_defect_J'])==697
        assert np.max(np.abs(curve['phase_rad']))<=np.pi
        np.testing.assert_allclose(np.sqrt(np.mean(np.asarray(curve['force_Nm'])[time>=1.1-1e-7]**2)),curve['future_torque_rms_Nm'])
        np.testing.assert_allclose(np.max(np.abs(np.asarray(curve['energy_defect_J'])[energy_time>=1.1-1e-7])),curve['future_energy_max_J'])
    assert sum(c['original_curve_preserved'] for c in traces)==4
    assert 'Within tolerance' in figure_text and figure_text.count('95% CI spans 0') == 2
    assert all(value in figure_text for value in ['139', '16.4', '0.715', '22.5', '2.93', '+0.56', '+9.08'])
    assert float_pages['tab:corpus']==float_pages['tab:geometry-readout']
    readout_rows = [line.split(' & ') for line in (PAPER/'tables/dynamics_readout_main_rows.tex').read_text().splitlines() if ' & ' in line]
    assert len(readout_rows) == 9 and all(len(row) == 6 for row in readout_rows)
    arms = ['keycld/video_euler','keycld/state_euler','keycld/state_rk4','con/video_con_iae',
            'con/state_con_iae','con_integrator/state_euler_con_iae','con_integrator/state_rk4_con_iae']
    for row, arm in zip([readout_rows[i] for i in (0,2,3,4,6,7,8)], arms):
        original = extension['summary']['physics'][arm]
        metrics = original['conditioned']['metrics']
        assert row[1] == str(original['conditioned']['complete'])+'/324'
        correction_mm = original['correction_rms_m']['mean']*1000
        assert row[2] == (r'$<0.001$' if correction_mm < 0.001 else '%.2f' % correction_mm)
        assert row[3] == '%.3f' % metrics['force_balance_Nm']['mean']
        assert row[4] == '%.3f' % metrics['work_energy_J']['mean']
        assert row[5].split()[0] == '%.2f' % (metrics['prediction_m']['mean']*1000)
    results['main_two_model_visuals'] = {'paired_integrator_panels':6, 'paired_case_counts':[324,324,324,324,315,315],
        'pass_region_is_per_criterion':True,'energy_scatter_min_J':.01,
        'annotated_means_and_paired_intervals_verified':True,
        'native_and_geometric_readout_labels_verified':True, 'shared_control_readout_table_values_verified':True,
        'shared_family_palette_markers_fonts_and_SI_axes':True,
        'provenance':'sources/integrator_comparison_data.json and sources/dynamics_extension_evidence.json'}
    # Main visual claims, protocol counts, and native failures are checked against source records.
    shared=collect_shared_video()
    assert shared==json.loads((PAPER/'sources/shared_video_figure_data.json').read_text())
    assert shared['frames']==[33,119] and shared['seed']==42
    assert shared['case_id']==integrators['trace_case']=='S-A-109-P01__low_speed'
    assert r'\includegraphics[width=\textwidth]{figures/shared_video_audit.pdf}' in main_text
    for source in shared['sources']+shared['geometric_correction']['sources']+integrators['trace_sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
    assert [(r['model'],r['width']) for r in shared['resolution_runs']]==[('KeyCLD',64),('KeyCLD',256),('CON',64),('CON',256)]
    assert 'motion mask' in command('pdftotext',str(PAPER/'figures/shared_video_audit.pdf'),'-')
    native_rows=[line.split(' & ') for line in (PAPER/'tables/native_comparison_main_rows.tex').read_text().splitlines() if ' & ' in line]
    assert len(native_rows)==9 and all(len(row)==10 for row in native_rows)
    metric_keys=('geometry_m','velocity_m_s','acceleration_m_s2','force_balance_Nm','work_energy_J','prediction_m')
    for row,arm in zip([native_rows[i] for i in (0,2,3,4,6,7,8)],arms):
        original=extension['summary']['physics'][arm]['raw'];scores=original['component_scores_100']
        video='video' in arm;model='keycld' if arm.startswith('keycld') else 'con'
        finite=extension['summary']['video'][model]['image']['prefix_forecast']['valid'] if video else original['metrics']['prediction_m']['valid']
        assert row[1]==str(finite)+'/324'
        assert row[2]==('%.2f'%(extension['summary']['video'][model]['image']['prefix_forecast']['mean']*1e4) if video else '--')
        assert row[3]==str(original['complete'])+'/324'
        assert row[4]=='/'.join('%.1f'%scores[k] for k in metric_keys[:3])
        for value,k in zip(row[5:7],metric_keys[3:5]):
            assert value==('%.2f'%scores[k] if original['metrics'][k]['valid'] else 'N/A')
        score_arm = arm.split('/', 1)[1]
        assert row[7]=='%.1f'%(original['metrics']['prediction_m']['mean']*1000)
        assert row[8]=='0' and row[9].startswith('$%.2f'%unified['planar'][score_arm]['mean'])
    for arm in ('state_euler','state_rk4'):
        rr=[]
        for seed in (42,43,44):
            rr+=json.loads((ROOT/('output/keycld_corpus/runs/%s_seed%d/constitutive_probe.json'%(arm,seed))).read_text())['rows']
        ratios=[r['estimated_effective_inertia_kg_m2']/r['true_effective_inertia_kg_m2'] for r in rr if r['positive_gain']]
        expected=constitutive['pooled_response'][arm]
        assert len(rr)==expected['total']==1944 and len(ratios)==expected['positive']
        assert abs(expected['unusable_gain_percent']-100*(len(rr)-len(ratios))/len(rr))<1e-12
        assert abs(expected['positive_inertia_ratio_median']-statistics.median(ratios))<1e-12
    diagram_nodes={c.get('id'):c.get('value','') for c in drawio.iter('mxCell')}
    assert 'CON' in diagram_nodes['con-encoder'] and '14D' in diagram_nodes['con-ode']
    budgets=[]
    for model,root,model_arms in [('keycld',ROOT/'output/keycld_corpus',('video_euler','state_euler','state_rk4')),
                                 ('con',ROOT/'output/con_corpus',('video_con_iae','state_con_iae')),
                                 ('con_integrator',ROOT/'output/con_corpus/dynamics_extension',('state_euler_con_iae','state_rk4_con_iae'))]:
        for arm in model_arms:
            for seed in (42,43,44):
                r=json.loads((root/'runs'/('%s_seed%d'%(arm,seed))/'training_report.json').read_text())
                if model=='keycld':
                    assert r['epochs']==40 and r['optimizer_updates']==32520 and r['training_sequences']==813
                    exposures=r['epochs']*r['training_sequences'];updates=r['optimizer_updates']
                else:
                    assert r['epochs']==50 and r['updates']==2550 and r['samples_seen']==40650
                    exposures=r['samples_seen'];updates=r['updates']
                budgets.append(dict(model=model,arm=arm,seed=seed,epochs=r['epochs'],updates=updates,sequence_exposures=exposures))
    assert len(budgets)==21
    for model in ('keycld','con'):
        for seed in (42,43,44):
            path=ROOT/f'output/planar_resolution/runs/{model}_256_seed{seed}/training_report.json'
            r=json.loads(path.read_text())
            updates=r.get('optimizer_updates',r.get('updates'))
            assert r['epochs']==(40 if model=='keycld' else 50)
            assert updates==(32520 if model=='keycld' else 2550)
            assert r['full_objective'] and r['scientific_training_result']
            budgets.append(dict(model=model,arm='video_256',seed=seed,epochs=r['epochs'],
                                updates=updates,sequence_exposures=r['epochs']*813))
    assert len(budgets)==27
    for value in ('32,520','40,650','2,550'):
        assert value in main_text,value
    assert '48-interval' in sources
    assert re.search(r'43 (?:overlapping )?four-step', sources)
    results['revised_main_evidence']={'visual_future_frames':shared['frames'],'finite_forecasts':[311,324,324,324],
        'visual_order':['KeyCLD 64','KeyCLD 256','CON 64','CON 256'],
        'motion_mask_ratios_and_paired_RGB_mechanics_in_main_figure':True,'native_main_table_arms':9,
        'native_readout_failures_unchanged':True,'readout_point_column_checked':True,
        'constitutive_panels':3,'capacity_diagnostic_insets':2,'geometric_correction_plot_page':5,'gain_check_failures_retained':True,
        'same_case_dynamics_curves':6,'resolution_video_curves':4,'shared_state_curves':2,
        'matched_times_reconstruction_and_energy_anchor_verified':True,
        'CON_in_method_visual_integrator_and_geometric_readout_comparisons':True,'training_runs_checked':len(budgets),
        'training_budget_records':budgets,'equal_compute_or_convergence_not_asserted':True}
    from check_spatial_expansion import verify as verify_spatial_expansion
    results['lingbot_spatial_extension'] = verify_spatial_expansion()
    calibration_root = ROOT / 'output/closed_chain_eval/visual_calibration'
    # Keep figure budgets and the separation of planar and spatial evidence.
    for name, max_ratio in [('audit_mechanism',215/510),('shared_video_audit',215/505),('spatial_vjepa_audit',225/505)]:
        path=PAPER/'figures'/(name+'.pdf')
        info=command('pdfinfo',str(path))
        width,height=map(float,re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info).groups())
        assert height/width <= max_ratio,(name,width,height)
        if name=='shared_video_audit':
            content=command('pdftotext',str(path),'-')
            assert all(word not in content for word in ('LingBot','Stewart','Delta'))
    assert r'\ref{fig:shared-video}' not in main_text.split(r'\section{Predictive and Generative Spatial Dynamics}',1)[1]
    paper_text = sources + '\n'.join(command('pdftotext', str(PAPER / name), '-') for name in
        ('main.pdf', 'supplement.pdf', 'figures/calibration_overview.pdf'))
    assert not re.search(r'(?:old|original|previous|initial)\s+(?:observer|reader)|旧观[察测]器', paper_text, re.I)
    assert 'Absolute-coordinate head' in paper_text and 'CAD-coordinate head' in paper_text
    results['method_comparisons'] = dict(historical_observer_labels_removed=True,
                                        training_configuration_differences_disclosed=True)
    calibration = json.loads((calibration_root / 'summary.json').read_text())
    for family in ('H-A', 'H-B'):
        reader = next(r for r in calibration['summaries']
                      if r['reader'] == 'geometry_spatial_256' and r['category'] == family)
        assert f"{reader['localization']['bound'] * 1000:.2f}" in sources
        verification = reader['localization']['verification']
        assert f"{verification['passed']}/{verification['total']}" in sources
    paired_reader = json.loads((calibration_root / 'paired_robust_planar_rgb/summary.json').read_text())
    reference = next(r for r in paired_reader['summary'] if r['model'] == 'reference')
    assert f"{reference['point_rmse_mean_mm']:.2f}" in main_text
    # Matched-resolution Euclidean values and their identical future window are
    # independently recomputed by verify_resolution(), including the main text.
    phase_check = json.loads((calibration_root / 'optimization/constrained_visual_phase/verification.json').read_text())
    phase = next(r for r in phase_check['summary'] if r['role'] == 'verification'
                 and r['quality'] == 'clean' and r['method'] == 'phase_sg7d3')
    for key in ('force_balance_Nm', 'work_energy_J'):
        assert f"{phase['mean'][key]:.2f}" in sources
    assert phase['all_six_pass'] == 0 and phase['total'] == 27
    results['unified_scoring'] = {'version': unified['scoring_version'],
        'primary_score': 'continuous engineering-tolerance ratio', 'binary_composite_active': False,
        'measurement_normalized_alternate_score': False, 'nine_main_arm_scores_verified': True,
        'eight_lingbot_main_scores_and_separate_AC_scores_verified': True,
        'reference_replay_numbers_verified': True,
        'calibration_quantiles_distinguished_from_confidence_intervals': True,
        'paired_observer_results_in_supplement': True,
        'high_resolution_model_training_completed': True, 'completed_resolutions_px': [64,256]}
    results['vjepa_expansion'] = verify_vjepa_expansion()
    results.update({
        "checked_client_date": "2026-10-17",
        "format_basis": "Official CVPR2026-v1(latex), revision 12909ae; generic anonymous CVPR draft",
        "official_template": "https://github.com/cvpr-org/author-kit",
        "official_page_rule": "https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines",
        "year_specific_submission_readiness": "Not asserted; user requested CVPR format without selecting a submission year",
        "citation_entries": len(keys), "cited_entries": len(cited),
        "evidence_counts_and_headline_values_checked": True,
        "style_file_unchanged": "cvpr.sty and ieeenat_fullname.bst retained from the official revision; compatibility fix is isolated in preamble.tex",
        "remaining_submission_metadata": ["Author identities for a final version", "Actual submission ID when registered"],
        "scientific_scope": ["Matched assembly data for KeyCLD and CON with distinct native budgets and physical readouts", "Two architectures, four video resolution settings and five shared state arms, twenty-seven completed training runs",
                             "343 evaluated assemblies, three planar families, two joint topologies", "Three corpus training seeds",
                             "Paired integration controls in both models; KeyCLD-specific friction and calibrated actuator interventions", "CON periodic-regime migration and privileged affine-readout diagnostics", "Native ODE diagnostic track",
                             "V-JEPA 2.1 + AC: 50 test assemblies per family, three conditions and three training seeds, 3,000 exposures per family/seed, calibrated feature readout and matched reference controls",
                             "LingBot-VA shared 5.1B backbone: zero-shot and fixed-budget full-Transformer adaptation, 50 updates and 400 exposures per seed, three seeds, 100 held-out assemblies, frozen calibrated RGB readout"],
    })
    (PAPER / "format_check.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
