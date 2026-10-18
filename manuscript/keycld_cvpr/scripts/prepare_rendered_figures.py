"""Paper figures from recorded Blender renders; no new simulation or model run."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
from matplotlib.path import Path as PlotPath
import matplotlib.patheffects as pe

from prepare_artifacts import ROOT, PAPER, BLUE, ORANGE, GREEN

ASSETS = PAPER / 'figures/assets'
FAMILIES = ('S-A', 'S-B', 'S-J')
NAMES = ('Crank-rocker · 4R', 'Offset crank-slider · 3R+1P', 'Double-crank · 4R')
COLORS = (BLUE, ORANGE, GREEN)
SPLITS = ('train', 'validation', 'calibration', 'test')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_frame(row, view='V09'):
    folder = (ROOT / row['physical_parameters']).parents[2]
    stem = row['assembly_id'] + '-' + view
    video = folder / 'videos' / (stem + '.mp4')
    metadata = json.loads(video.with_suffix('.json').read_text())
    assert metadata['assembly_id'] == row['assembly_id']
    assert metadata['condition'] == 'rated' and metadata['view'] == 'isometric'
    assert sha(video) == metadata['sha256'], str(video)
    cap = cv2.VideoCapture(str(video))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Missing recorded render: ' + str(video))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), video


def prepare_assets(manifest):
    cv2.setNumThreads(1)
    ASSETS.mkdir(exist_ok=True)
    records = sorted((r for r in manifest['records'] if r['condition'] == 'rated'),
                     key=lambda r: (FAMILIES.index(r['category']), SPLITS.index(r['split']), r['assembly_id']))
    assert len(records) == len({r['assembly_id'] for r in records}) == 343
    examples = {}
    for category in FAMILIES:
        test = [r for r in records if r['category'] == category and r['split'] == 'test']
        examples[category] = [test[0], test[len(test)//2], test[-1]]
    source_path = PAPER / 'sources/rendered_figure_sources.json'
    if source_path.exists():
        cached = json.loads(source_path.read_text())
        if ([r['assembly_id'] for r in cached['records']] == [r['assembly_id'] for r in records]
                and all((PAPER/r['asset']).exists() and sha(PAPER/r['asset']) == r['asset_sha256']
                        for r in cached['records'])
                and all((ASSETS/(r['assembly_id']+'_detail.png')).exists()
                        for group in examples.values() for r in group)):
            return records, examples
    provenance = []
    for row in records:
        image, source = source_frame(row)
        target = ASSETS / (row['assembly_id'] + '.png')
        image.resize((192, 192), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS).save(target)
        provenance.append({'assembly_id': row['assembly_id'], 'family': row['category'], 'split': row['split'],
                           'condition': 'rated', 'view': 'isometric', 'time_s': 0,
                           'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
                           'asset': str(target.relative_to(PAPER)), 'asset_sha256': sha(target)})
    examples = {}
    for category in FAMILIES:
        test = [r for r in records if r['category'] == category and r['split'] == 'test']
        # First, middle, and last test identities: independent of any model result.
        examples[category] = [test[0], test[len(test)//2], test[-1]]
        for row in examples[category]:
            image, _ = source_frame(row)
            image.save(ASSETS / (row['assembly_id'] + '_detail.png'))
    first = examples['S-A'][0]
    with np.load(ROOT / 'output/keycld_corpus' / first['inputs']) as z:
        for i, frame in enumerate(z['x'][:3]):
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            Image.fromarray(frame).save(ASSETS / ('observed_%d.png' % i))
    payload = {'description': 'All 343 participating assemblies, first decoded frame of each verified rated-condition isometric video.',
               'video_hashes_match_render_metadata': True,
               'time_s': 0, 'rows_order': 'family, train/validation/calibration/test, canonical assembly ID',
               'thumbnail_size_px': 192, 'counts': {c: sum(r['category'] == c for r in records) for c in FAMILIES},
               'detail_selection': 'First, middle and last held-out identity in each family; no score-based selection.',
               'examples': {c: [r['assembly_id'] for r in rows] for c, rows in examples.items()},
               'model_input': {'assembly_id': first['assembly_id'], 'source': first['inputs'],
                               'source_sha256': first['inputs_sha256'], 'frame_times_s': [1, 31/30, 32/30],
                               'size_px': 64, 'view': 'calibrated front'},
               'records': provenance}
    (PAPER / 'sources/rendered_figure_sources.json').write_text(json.dumps(payload, indent=2) + '\n')
    return records, examples


def export(fig, name, dpi=300):
    fig.savefig(PAPER / 'figures' / (name + '.pdf'), dpi=dpi, bbox_inches='tight', pad_inches=.02)
    fig.savefig(PAPER / 'figures' / (name + '.svg'), dpi=dpi, bbox_inches='tight', pad_inches=.02)
    fig.savefig(PAPER / 'build' / (name + '.png'), dpi=210, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def coverage(manifest):
    capacity_path=ROOT/'output/keycld_corpus/kinetic_capacity_audit.json'
    capacity=json.loads(capacity_path.read_text())
    rated={r['assembly_id']:r for r in capacity['rows'] if r['condition']=='rated'}
    rows=[];configs={}
    for record in manifest['records']:
        path=ROOT/record['physical_parameters']
        assert sha(path)==record['physical_parameters_sha256']
        physical=json.loads(path.read_text());cfg=physical['config']
        configs[(record['assembly_id'],record['condition'])]=physical
        if record['condition']!='rated':continue
        metric=np.asarray(rated[record['assembly_id']]['target_metric_kg_m2'])
        assert len(metric)==41
        rows.append({'assembly_id':record['assembly_id'],'family':record['category'],'split':record['split'],
                     'crank_length_mm':cfg['lengths_m'][0]*1000,
                     'coupler_crank_length_ratio':cfg['lengths_m'][1]/cfg['lengths_m'][0],
                     'moving_mass_kg':sum(body['mass_kg'] for body in physical['bodies'].values()),
                     'median_reduced_inertia_kg_m2':float(np.median(metric))})
    for r in rows:
        nominal=configs[(r['assembly_id'],'rated')]['config']['payload_mass_kg']
        for condition,speed,ratio in [('low_speed',1,1),('rated',3,1),('limit_or_load',5,1.75)]:
            p=configs[(r['assembly_id'],condition)]
            assert p['command_speed_rad_s']==speed
            np.testing.assert_allclose(p['config']['payload_mass_kg']/nominal,ratio)
    assert len(rows)==343
    result={'source':'output/keycld_corpus/dataset_manifest.json',
            'source_sha256':sha(ROOT/'output/keycld_corpus/dataset_manifest.json'),
            'physical_parameter_hashes_verified':1029,
            'inertia_source':str(capacity_path.relative_to(ROOT)),'inertia_source_sha256':sha(capacity_path),
            'assemblies':343,'splits':manifest['assemblies_by_split'],
            'sequence_counts':manifest['counts'],'poses_per_inertia_summary':41,
            'operating_conditions':{'speed_rad_s':[1,3,5],'payload_multiplier':[1,1,1.75]},
            'scope':'Rated-condition physical parameters; moving-body masses include mounted payload exactly once; median reduced inertia uses 41 recorded poses.',
            'rows':rows}
    (PAPER/'sources/dataset_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    return rows


def display_image(path, crop=None):
    """Uniform display contrast only; recorded assets and model inputs stay intact."""
    image = Image.open(path).convert('RGB')
    if crop is not None:
        image = image.crop(crop)
    rgb = np.asarray(image, dtype=float)/255
    return np.clip((rgb-.65)*1.6+.78, 0, 1)


def spatial_atlas_assets():
    """Recorded test-assembly examples for the separate spatial extension."""
    path=ROOT/'output/vjepa_expansion/dataset_manifest.json'
    manifest=json.loads(path.read_text())
    selected=[]
    for category,name in (('H-A','Stewart'),('H-B','Delta')):
        row=sorted((r for r in manifest['videos'] if r['category']==category
                    and r['split']=='test' and r['condition']=='rated'),
                   key=lambda r:r['assembly_id'])[0]
        # The atlas illustrates the registered assembly from its recorded
        # isometric camera; evaluation uses the manifest's front view.
        folder=(ROOT/row['video']).parent
        sidecars=sorted(folder.glob(row['assembly_id']+'-V*.inputs.json'))
        inputs=next(p for p in sidecars if
                    (json.loads(p.read_text())['condition'],
                     json.loads(p.read_text())['view'])==('rated','isometric'))
        video=inputs.with_name(inputs.name.replace('.inputs.json','.mp4'))
        cap=cv2.VideoCapture(str(video))
        ok,frame=cap.read();cap.release()
        assert ok,video
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        asset=ASSETS/(category+'_atlas.png')
        image.save(asset)
        family=[r for r in manifest['videos'] if r['category']==category]
        selected.append(dict(category=category,name=name,assembly_id=row['assembly_id'],
            video_id=video.stem,evaluation_video_id=row['video_id'],split=row['split'],
            condition=row['condition'],view='isometric',
            frame=0,time_s=0,source=str(video.relative_to(ROOT)),source_sha256=sha(video),
            public_inputs=str(inputs.relative_to(ROOT)),public_inputs_sha256=sha(inputs),
            asset=str(asset.relative_to(PAPER)),asset_sha256=sha(asset),
            assemblies=len({r['assembly_id'] for r in family}),
            test_identities=len({r['assembly_id'] for r in family if r['split']=='test'})))
    return dict(manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path),
                selection='First held-out identity per spatial family, rated-condition isometric frame 0.',
                scope='Separate spatial corpus; excluded from the 343 planar identities and panels b--d.',
                examples=selected)


def measured_test_dynamics(manifest):
    """Summarize saved physical references on the registered future interval."""
    rows=[]
    for record in sorted(manifest['records'],key=lambda r:r['case_id']):
        if record['split']!='test':
            continue
        physical_path=ROOT/record['physical_parameters']
        reference_path=ROOT/record['physics_reference']
        config=json.loads(physical_path.read_text())['config']
        limit=config['actuator']['torque_limit_Nm']
        with np.load(reference_path) as data:
            mask=(data['time']>=1.1-1e-9)&(data['time']<=119/30+1e-9)
            time=data['time'][mask]
            torque=data['actual_motor_torque_Nm'][mask]
            power=data['motor_power_W'][mask]
            np.testing.assert_allclose(power,torque*data['qdot'][mask,0],atol=1e-10)
            assert len(time)==689 and np.all(np.isfinite(torque)) and np.all(np.isfinite(power))
            rows.append({'case_id':record['case_id'],'assembly_id':record['assembly_id'],
                         'family':record['category'],'condition':record['condition'],
                         'torque_limit_Nm':limit,
                         'torque_rms_percent_limit':float(100*np.sqrt(np.mean(torque**2))/limit),
                         'absolute_motor_work_J':float(np.trapz(np.abs(power),time)),
                         'reference':str(reference_path.relative_to(ROOT)),
                         'reference_sha256':sha(reference_path),
                         'physical_parameters':str(physical_path.relative_to(ROOT)),
                         'physical_parameters_sha256':sha(physical_path)})
    conditions=('low_speed','rated','limit_or_load')
    assert len(rows)==108 and len({r['assembly_id'] for r in rows})==36
    summaries={}
    for condition in conditions:
        selected=[r for r in rows if r['condition']==condition]
        assert len(selected)==36 and all(sum(r['family']==f for r in selected)==12 for f in FAMILIES)
        summaries[condition]={metric:dict(zip(('min','q25','median','q75','max'),
                            np.percentile([r[metric] for r in selected],[0,25,50,75,100]).tolist()))
                            for metric in ('torque_rms_percent_limit','absolute_motor_work_J')}
    evidence={'selection':'All 36 registered test identities, three conditions each; saved simulator references.',
              'interval_s':[1.1,119/30],'samples_per_case':689,'reference_rate_Hz':240,
              'torque_definition':'100 * RMS(actual motor torque) / public actuator torque limit',
              'work_definition':'Trapezoidal integral of absolute motor power; an excitation statistic, not signed net work or an energy defect.',
              'aggregation':'One value per reference trajectory; each condition contains 12 identities per family.',
              'rows':rows,'summary':summaries}
    (PAPER/'sources/operating_dynamics.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def motion_overlay(ax, record):
    """Place schematic motion arrows using the recorded pose and calibrated camera."""
    folder = (ROOT/record['physical_parameters']).parents[2]
    metadata_path = folder/'videos'/(record['assembly_id']+'-V09.json')
    poses_path = folder/'conditions/rated/render_poses.npz'
    physical_path = ROOT/record['physical_parameters']
    metadata = json.loads(metadata_path.read_text())
    physical = json.loads(physical_path.read_text())
    config = physical['config']
    camera = metadata['camera_frames'][0]
    rotation = np.asarray(camera['R_world_to_camera'])
    translation = np.asarray(camera['t_world_to_camera_m'])
    intrinsic = np.asarray(camera['K_affine_pixels'])
    with np.load(poses_path) as data:
        names = list(data['body_names'])
        positions = dict(zip(names, data['body_position_m'][0]))
        quaternions = dict(zip(names, data['body_quaternion_wxyz'][0]))

    def project(points):
        camera_points = np.atleast_2d(points)@rotation.T+translation
        return camera_points[:, :2]@intrinsic[:2, :2].T+intrinsic[:2, 2]

    def arrow(points, double=False):
        patch = FancyArrowPatch(path=PlotPath(project(points)),
                                arrowstyle='<->' if double else '-|>',
                                mutation_scale=5.5, lw=.85, color='#172E40', zorder=4)
        patch.set_path_effects([pe.Stroke(linewidth=1.8, foreground='white'), pe.Normal()])
        ax.add_patch(patch)

    def arc(center, radius, start, end, double=False):
        angles = np.linspace(start, end, 40)
        points = np.asarray(center)+radius*np.c_[np.cos(angles), np.zeros(40), np.sin(angles)]
        arrow(points, double)

    crank = positions['crank'].copy()
    crank[1] = config['visual_and_com_y_m'][0]-.025
    arc(crank, .052, -.35*np.pi, 1.35*np.pi)
    if record['category'] == 'S-B':
        slider = positions['slider'].copy()
        slider[1] += config['visual_and_com_y_m'][2]
        slider[2] += .055
        arrow([slider+[-.07, 0, 0], slider+[.07, 0, 0]], double=True)
    else:
        rocker = positions['rocker'].copy()
        rocker[1] = config['visual_and_com_y_m'][2]-.025
        if record['category'] == 'S-A':
            arc(rocker, .075, .25*np.pi, .75*np.pi, double=True)
        else:
            arc(rocker, .058, -.35*np.pi, 1.35*np.pi)
    payload = next(part for part in physical['parts'] if part['name'] == 'removable_payload')
    w, x, y, z = quaternions['coupler']
    rot = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    payload_px = project(positions['coupler']+rot@payload['position_m'])[0]
    ax.annotate('Payload', xy=payload_px, xytext=(.71, .94), textcoords='axes fraction',
                fontsize=5.5, ha='center', va='top', color='#172E40', zorder=5,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': .2},
                arrowprops={'arrowstyle': '-', 'color': '#172E40', 'lw': .55, 'shrinkB': 2})
    return {'assembly_id': record['assembly_id'], 'payload_pixel': payload_px.tolist(),
            'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
                        for p in (metadata_path, poses_path, physical_path)]}


def density_background(ax, rows, xkey, ykey, xlim, ylim, log_y=False, log_x=False):
    """Pooled sample density in normalized plotting coordinates, not family labels."""
    x = np.asarray([r[xkey] for r in rows])
    y = np.asarray([r[ykey] for r in rows])
    x_bounds = np.log10(xlim) if log_x else np.asarray(xlim)
    x_values = np.log10(x) if log_x else x
    y_bounds = np.log10(ylim) if log_y else np.asarray(ylim)
    y_values = np.log10(y) if log_y else y
    x_normalized = (x_values-x_bounds[0])/(x_bounds[1]-x_bounds[0])
    y_normalized = (y_values-y_bounds[0])/(y_bounds[1]-y_bounds[0])
    grid = np.linspace(0, 1, 120)
    bandwidth = (.045, .055)
    kx = np.exp(-.5*((grid[None, :]-x_normalized[:, None])/bandwidth[0])**2)
    ky = np.exp(-.5*((grid[None, :]-y_normalized[:, None])/bandwidth[1])**2)
    density = ky.T @ kx
    density /= density.max()
    gx = x_bounds[0]+grid*(x_bounds[1]-x_bounds[0])
    gy = y_bounds[0]+grid*(y_bounds[1]-y_bounds[0])
    if log_y:
        gy = 10**gy
    if log_x:
        gx = 10**gx
    levels = [.08, .24, .50, .80, 1.001]
    ax.contourf(gx, gy, density, levels=levels,
                colors=['#F4F7FA', '#E8F0F6', '#DCE9F2', '#CDDFEC'], zorder=0)
    ax.contour(gx, gy, density, levels=levels[:-1], colors='#B7CBD9',
               linewidths=.35, alpha=.55, zorder=.5)
    return {'samples': len(rows), 'method': 'Gaussian kernel sum, pooled over families',
            'coordinates': ('log10 x and y' if log_x and log_y else
                            'log10 x; linear y' if log_x else
                            'linear x; log10 y' if log_y else 'linear x and y'),
            'normalized_axis_bandwidth': list(bandwidth), 'relative_peak_levels': levels,
            'xlim': list(xlim), 'ylim': list(ylim), 'grid': [120, 120]}


def overview(manifest):
    records, examples = prepare_assets(manifest)
    physical=coverage(manifest)
    spatial_examples=spatial_atlas_assets()
    fig = plt.figure(figsize=(7.05, 3.03))
    fig.text(.006,.992,'(a) Assembly atlas',ha='left',va='top',fontsize=8.5,weight='bold')
    annotation_sources = []
    atlas_ids = []
    for ci, (category, title, color) in enumerate(zip(FAMILIES, NAMES, COLORS)):
        left = .006 + ci*.149
        width = .143
        count=sum(r['category']==category for r in records)
        topology='3R+1P' if category=='S-B' else '4R'
        fig.text(left+width/2,.93,'%s · %s · %d'%(category,topology,count),ha='center',fontsize=6.8,color=color)
        first = examples[category][0]
        ax = fig.add_axes([left,.71,width,.205])
        crop = (56, 64, 600, 404)
        ax.imshow(display_image(ASSETS/(first['assembly_id']+'_detail.png'), crop),
                  extent=(crop[0],crop[2],crop[3],crop[1]), interpolation='none')
        ax.set_xlim(crop[0],crop[2]); ax.set_ylim(crop[3],crop[1]); ax.set_axis_off()
        annotation_sources.append(motion_overlay(ax, first))
        rows = [r for r in records if r['category'] == category and r['split'] == 'test']
        assert len(rows) == 12
        atlas_ids.extend(r['assembly_id'] for r in rows)
        ax = fig.add_axes([left,.421,width,.252])
        cols = 4
        ax.set_xlim(0, cols); ax.set_ylim(3, 0); ax.set_aspect('equal'); ax.axis('off')
        for i, row in enumerate(rows):
            x, y = i % cols, i // cols
            ax.imshow(display_image(ASSETS / (row['assembly_id'] + '.png')), extent=(x+.035,x+.965,y+.965,y+.035),
                      interpolation='none', zorder=1)
            ax.add_patch(Rectangle((x+.012,y+.012), .976,.976,fill=False,edgecolor='#B5C2CA',lw=.3,zorder=3))
    fig.text(.006,.681,'36 planar test identities · 12 per family',fontsize=6.6,color='#536775')
    fig.text(.006,.389,'Spatial extension · Stewart and Delta',fontsize=6.6,color='#536775')
    for i,record in enumerate(spatial_examples['examples']):
        center=.120+i*.225
        fig.text(center,.355,f"{record['category']} · {record['name']} · {record['test_identities']}",
                 ha='center',fontsize=6.8,color=('#176B78','#B45D38')[i])
        ax=fig.add_axes([center-.071,.014,.142,.326])
        ax.imshow(display_image(PAPER/record['asset'],(68,52,572,612)),interpolation='none')
        ax.set_axis_off()

    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',color=color,ls='',markersize=3.5,label=name.split(' · ')[0])
             for color,name in zip(COLORS,NAMES)]
    fig.legend(handles=handles,bbox_to_anchor=(.516,.992),loc='upper left',frameon=False,
               ncol=3,fontsize=6.4,handlelength=.6,handletextpad=.3,columnspacing=.8,borderaxespad=0)
    geom=fig.add_axes([.535,.54,.185,.335]);dynamics=fig.add_axes([.799,.54,.185,.335])
    density_specs = {
        'geometry': density_background(geom, physical, 'crank_length_mm', 'coupler_crank_length_ratio',
                                       (55,320), (.55,4.05)),
        'si_variation': density_background(dynamics, physical, 'moving_mass_kg', 'median_reduced_inertia_kg_m2',
                                           (.78,1.53), (.0018,.052), log_y=True)}
    for family,color in zip(FAMILIES,COLORS):
        rows=[r for r in physical if r['family']==family]
        test=[r for r in rows if r['split']=='test']
        for ax,xkey,ykey in [(geom,'crank_length_mm','coupler_crank_length_ratio'),
                            (dynamics,'moving_mass_kg','median_reduced_inertia_kg_m2')]:
            ax.scatter([r[xkey] for r in rows],[r[ykey] for r in rows],s=7,c=color,alpha=.55,lw=0,zorder=3)
            ax.scatter([r[xkey] for r in test],[r[ykey] for r in test],s=13,facecolors='none',edgecolors='#243747',lw=.5,zorder=4)
    geom.set(xlim=(55,320),ylim=(.55,4.05),xticks=[100,200,300],yticks=[1,2,3,4])
    geom.set_title('(b) Geometry',loc='left',fontsize=8.5,weight='bold',pad=7)
    geom.set_xlabel('Crank length (mm)',fontsize=7,labelpad=2)
    geom.set_ylabel('Coupler / crank length',fontsize=7,labelpad=2)
    dynamics.set(yscale='log',ylim=(.0018,.052),xlim=(.78,1.53),xticks=[.8,1.1,1.4],yticks=[.003,.01,.03])
    dynamics.set_yticklabels(['0.003','0.01','0.03'])
    dynamics.minorticks_off()
    dynamics.set_title('(c) SI variation',loc='left',fontsize=8.5,weight='bold',pad=7)
    dynamics.set_xlabel('Moving mass (kg)',fontsize=7,labelpad=2)
    dynamics.set_ylabel(r'Median inertia (kg m$^2$)',fontsize=7,labelpad=1)
    for ax in (geom,dynamics):ax.tick_params(labelsize=6.5,pad=2);ax.grid(color='#E1E7EC',lw=.45);ax.set_axisbelow(True)
    # Magnify the overlapping upper-left core without covering either point cloud.
    zoom_xlim, zoom_ylim = (76,128), (3.00,3.68)
    zoom_box = (.53,.36,.45,.49)
    zoom = geom.inset_axes(zoom_box, zorder=6)
    for family,color in zip(FAMILIES,COLORS):
        rows = [r for r in physical if r['family']==family]
        test = [r for r in rows if r['split']=='test']
        for group, size, style in [(rows,5,{'c':color,'alpha':.60,'lw':0}),
                                    (test,9,{'facecolors':'none','edgecolors':'#243747','lw':.5})]:
            zoom.scatter([r['crank_length_mm'] for r in group],
                         [r['coupler_crank_length_ratio'] for r in group],s=size,zorder=3,**style)
    zoom.set(xlim=zoom_xlim,ylim=zoom_ylim,xticks=[80,120],yticks=[3.2,3.6])
    zoom.set_title('Zoom',fontsize=5.8,pad=2,loc='right',color='#536775')
    zoom.tick_params(labelsize=5.5,length=2,pad=1)
    zoom.grid(color='#E5EBEF',lw=.35);zoom.set_axisbelow(True)
    for spine in zoom.spines.values():
        spine.set_visible(True);spine.set_color('#718895');spine.set_linewidth(.6)
    geom.add_patch(Rectangle((zoom_xlim[0],zoom_ylim[0]),np.ptp(zoom_xlim),np.ptp(zoom_ylim),
                            fill=False,edgecolor='#718895',lw=.65,zorder=5))
    for y,corner in zip(zoom_ylim,[(0,0),(0,1)]):
        geom.add_artist(ConnectionPatch(xyA=(zoom_xlim[1],y),coordsA=geom.transData,
                                       xyB=corner,coordsB=zoom.transAxes,
                                       color='#718895',lw=.55,zorder=2))
    measured=measured_test_dynamics(manifest)
    fig.text(.516,.385,'(d) Measured test dynamics',fontsize=8.5,weight='bold')
    conditions=('low_speed','rated','limit_or_load')
    for left,metric,title,ylabel in [
            (.535,'torque_rms_percent_limit','Actuator demand','RMS torque (% limit)'),
            (.799,'absolute_motor_work_J','Actuator work',r'$\int |\tau_A\omega|\,dt$ (J)')]:
        ax=fig.add_axes([left,.115,.185,.205])
        grouped=[[r[metric] for r in measured['rows'] if r['condition']==c] for c in conditions]
        ax.boxplot(grouped,positions=range(3),widths=.56,whis=(0,100),showfliers=False,
                   patch_artist=True,manage_ticks=False,zorder=1,
                   boxprops={'facecolor':'#F2F5F7','edgecolor':'#81929D','linewidth':.5},
                   medianprops={'color':'#243747','linewidth':1.0},
                   whiskerprops={'color':'#81929D','linewidth':.5},
                   capprops={'color':'#81929D','linewidth':.5})
        for ci,condition in enumerate(conditions):
            for fi,(family,color) in enumerate(zip(FAMILIES,COLORS)):
                group=[r for r in measured['rows'] if r['condition']==condition and r['family']==family]
                jitter=np.linspace(-.055,.055,len(group))+(fi-1)*.12
                ax.scatter(ci+jitter,[r[metric] for r in group],s=7,c=color,alpha=.75,lw=0,zorder=3)
        ax.set(xlim=(-.5,2.5),xticks=range(3),xticklabels=['Low\n1 rad/s','Rated\n3 rad/s','Load\n5 rad/s\n1.75× payload'])
        if metric=='torque_rms_percent_limit':
            ax.set(ylim=(0,18),yticks=[0,5,10,15])
        else:
            ax.set(yscale='log',ylim=(.1,30),yticks=[.1,1,10])
            ax.set_yticklabels(['0.1','1','10']);ax.minorticks_off()
        ax.set_title(title,loc='left',fontsize=6.8,pad=4)
        ax.set_ylabel(ylabel,fontsize=6.5,labelpad=1.5)
        ax.tick_params(labelsize=6.1,pad=2,length=2)
        ax.grid(axis='y',color='#E1E7EC',lw=.45);ax.set_axisbelow(True)
    export(fig, 'corpus_overview', dpi=360)
    provenance_path = PAPER/'sources/rendered_figure_sources.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['atlas_display'] = {'identities': atlas_ids, 'count': 36, 'grid_per_family': [3,4],
                                   'enlarged_examples': 3, 'full_corpus_in_scatter_panels': 343,
                                   'spatial_extension': spatial_examples,
                                   'spatial_crop_xyxy_pixels':[68,52,572,612],
                                   'primary_crop_xyxy_pixels': list(crop),
                                   'contrast': {'operation': 'clip((RGB/255 - 0.65)*1.6 + 0.78, 0, 1)',
                                                'scope': 'Figure 2 display only; original render assets unchanged'},
                                   'motion_arrows': 'Schematic rotation/reciprocation at calibrated body locations',
                                   'motion_annotation_sources': annotation_sources}
    provenance['density_backgrounds'] = density_specs
    provenance.pop('sequence_protocol_display',None)
    provenance['operating_condition_display'] = {
        'layout':'Two measured distributions in the lower-right panel; horizontal family legend',
        'source':'sources/operating_dynamics.json','test_trajectories':108,
        'per_condition':36,'speed_rad_s':[1,3,5],'payload_multiplier':[1,1,1.75],
        'boxplot':'Median, interquartile box, full-range whiskers; every trajectory is shown.'}
    provenance['geometry_zoom'] = {'xlim':list(zoom_xlim),'ylim':list(zoom_ylim),
                                   'axes_bounds':list(zoom_box),'all_parent_points_retained':True,
                                   'identities':[r['assembly_id'] for r in physical
                                                if zoom_xlim[0]<=r['crank_length_mm']<=zoom_xlim[1]
                                                and zoom_ylim[0]<=r['coupler_crank_length_ratio']<=zoom_ylim[1]]}
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    overview(json.loads((ROOT / 'output/keycld_corpus/dataset_manifest.json').read_text()))
