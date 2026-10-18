"""Generate deterministic appearance variants from existing audited trajectories.

New mechanism dimensions require new physics inputs; this command varies only
appearance. Each Blender process is released before the next job starts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['nominal'])
    parser.add_argument('--styles', nargs='+', default=['industrial', 'graphite', 'light'])
    parser.add_argument('--input-root', type=Path, default=ROOT/'output/fourbar')
    parser.add_argument('--output-root', type=Path, default=ROOT/'output/fourbar/appearance_variants')
    parser.add_argument('--appearance', type=Path, default=SOURCE/'appearance.json')
    parser.add_argument('--blender', default=os.environ.get('BLENDER_BIN'))
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--beauty', action='store_true')
    args = parser.parse_args()
    profile = json.loads(args.appearance.read_text())
    for style in args.styles:
        if style not in profile['styles']:
            parser.error('Unknown appearance style: '+style)
    # Case/style names are single directory names, never implicit filesystem paths.
    for name in [*args.cases, *args.styles]:
        if Path(name).name != name or name in ('.', '..'):
            parser.error('Case/style names must be single directory names')
    blender = args.blender or shutil.which('blender')
    if not blender:
        parser.error('Set BLENDER_BIN or --blender to an installed Blender executable')
    if args.animation and not shutil.which('ffmpeg'):
        parser.error('Animation encoding requires ffmpeg')
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['TMPDIR'] = str((ROOT/'output/fourbar/.tmp').resolve())
    environment['XDG_CACHE_HOME'] = str((ROOT/'output/fourbar/.cache').resolve())
    for key in ('TMPDIR', 'XDG_CACHE_HOME'):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    entries = []
    for case in dict.fromkeys(args.cases):
        source = (args.input_root/case).resolve()
        validation = json.loads((source/'validation.json').read_text())
        if not validation['passed']:
            raise RuntimeError('Input physics validation failed: '+case)
        for style in dict.fromkeys(args.styles):
            output = (args.output_root/case/style).resolve()
            if output == source:
                raise ValueError('Batch variants must have their own output directory')
            output.mkdir(parents=True, exist_ok=True)
            command = [blender, '-b', '--factory-startup', '--python', str(SOURCE/'blender_scene.py'), '--',
                       '--input-dir', str(source), '--output-dir', str(output),
                       '--appearance', str(args.appearance.resolve()), '--style', style,
                       '--render' if args.animation else '--preview-only']
            if args.beauty:
                command.append('--beauty')
            print('Generating', case, style, flush=True)
            with (output/'render.log').open('w') as log:
                subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
            # Blender can exit 0 after a Python script error; require fresh artifacts.
            log_text=(output/'render.log').read_text()
            if 'Traceback' in log_text or 'Error:' in log_text:
                raise RuntimeError('Blender reported an error; inspect '+str(output/'render.log'))
            appearance_manifest=json.loads((output/'appearance_manifest.json').read_text())
            if not appearance_manifest['checks']['animation_matches_recorded_states']:
                raise RuntimeError('Rendered states differ from physics inputs')
            for name,digest in appearance_manifest['builder_source_sha256'].items():
                if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Builder changed during the batch: '+name)
            filenames=['fourbar.blend','blender_preview.png','joint_detail.png','payload_detail.png',
                       'blender_camera.json','appearance_manifest.json']
            if args.beauty:
                filenames.append('assembly_beauty.png')
            if args.animation:
                config = json.loads((source/'physical_metadata.json').read_text())
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'warning', '-framerate',
                                str(config['simulation']['video_fps']), '-start_number', '1',
                                '-i', str(output/'blender_frames/frame_%04d.png'), '-c:v', 'libx264',
                                '-crf', '18', '-pix_fmt', 'yuv420p', str(output/'blender_rgb.mp4')], check=True)
                filenames.append('blender_rgb.mp4')
            entries.append({'case':case,'style':style,'directory':str(output.relative_to(args.output_root.resolve())),
                            'physics_source':str(source),'physics_input_sha256':appearance_manifest['physics_input_sha256'],
                            'builder_source_sha256':appearance_manifest['builder_source_sha256'],
                            'artifact_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in filenames}})
    (args.output_root/'batch_manifest.json').write_text(json.dumps({'schema_version':1,'scope':'appearance variants of existing validated trajectories','entries':entries},indent=2)+'\n')
    print('Completed',len(entries),'variants:',args.output_root/'batch_manifest.json')


if __name__=='__main__':
    main()
