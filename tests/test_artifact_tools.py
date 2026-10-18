"""Check corrupt downloads, restart behavior and safe bundle import."""
import hashlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


download=module('download_models')
artifacts=module('artifacts')
bootstrap=module('bootstrap')


class ArtifactTransport(unittest.TestCase):
    def test_portable_package_roundtrip_and_neutral_archive_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=root/'package';target=root/'imported';target.mkdir()
            source=package/'output/model/weight.bin';source.parent.mkdir(parents=True)
            data=b'portable model parameters';source.write_bytes(data)
            metadata=dict(title='Model',files=[dict(path='output/model/weight.bin',bytes=len(data))])
            (package/'package.json').write_text(json.dumps(metadata))
            (package/'README.md').write_text('Model and experiment files.\n')
            self.assertEqual(bootstrap.artifact_inputs(package),[package])
            self.assertEqual(artifacts.import_tree(package,target),1)
            self.assertEqual(artifacts.import_tree(package,target),1)
            archive=root/'model.tar'
            self.assertEqual(artifacts.export_package(package,archive),1)
            with tarfile.open(archive) as f:
                self.assertNotIn('artifact_manifest.json',f.getnames())
                self.assertNotIn(b'sha256',f.extractfile('package.json').read())
                for entry in f.getmembers():
                    self.assertEqual((entry.uid,entry.gid,entry.mtime),(0,0,0))
                    self.assertEqual((entry.uname,entry.gname),('',''))
            self.assertEqual(artifacts.import_bundle(archive,target),1)
            (target/'output/model/weight.bin').write_bytes(b'x'*len(data))
            with self.assertRaises(FileExistsError):artifacts.import_bundle(archive,target)
            with self.assertRaises(FileExistsError):artifacts.import_tree(package,target)

    def test_collection_verification_includes_optional_optimizer_states(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('current','pretrained','resume'):
                (root/name).mkdir();(root/name/'artifact_manifest.json').write_text('{"files": []}')
            index=root/'artifact_collection.json'
            index.write_text(json.dumps(dict(packages=[dict(path='current'),dict(path='pretrained'),dict(path='resume',default=False)])))
            self.assertEqual(artifacts.collection_packages(root),[root/'current',root/'pretrained'])
            self.assertEqual(artifacts.collection_packages(root,True),[root/'current',root/'pretrained',root/'resume'])
            index.write_text(json.dumps(dict(packages=[dict(path='current'),dict(path='current')])))
            with self.assertRaises(ValueError):artifacts.collection_packages(root,True)
            index.write_text(json.dumps(dict(packages=[dict(path='../outside',default=False)])))
            with self.assertRaises(ValueError):artifacts.collection_packages(root)

    def test_collection_selects_defaults_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);collection=root/'artifact_collection.json'
            for name in ('current','pretrained','resume'):
                (root/name).mkdir();(root/name/'artifact_manifest.json').write_text('{}')
            collection.write_text(json.dumps(dict(packages=[dict(path='current'),dict(path='pretrained'),dict(path='resume',default=False)])))
            self.assertEqual(bootstrap.artifact_inputs(root),[root/'current',root/'pretrained'])
            collection.write_text(json.dumps(dict(packages=[dict(path='../outside')])))
            with self.assertRaises(ValueError):bootstrap.artifact_inputs(root)

    def test_directory_import_preserves_data_and_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);tree=root/'tree';target=root/'target';target.mkdir()
            source=tree/'output/checkpoint.bin';source.parent.mkdir(parents=True)
            payload=b'complete checkpoint';source.write_bytes(payload)
            record=dict(path='output/checkpoint.bin',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            manifest=tree/'artifact_manifest.json';manifest.write_text(json.dumps(dict(files=[record])))
            self.assertEqual(artifacts.import_tree(tree,target,True),1)
            self.assertEqual(artifacts.import_tree(tree,target),1)
            self.assertEqual((target/record['path']).read_bytes(),payload)
            self.assertEqual(artifacts.import_tree(tree,target),1)
            source.write_bytes(b'x'*len(payload))
            with self.assertRaises(ValueError):artifacts.import_tree(tree,target,True)
            source.unlink();outside=root/'outside';outside.write_bytes(payload);source.symlink_to(outside)
            with self.assertRaises(ValueError):artifacts.import_tree(tree,target,True)
            manifest.write_text(json.dumps(dict(files=[record,record])))
            with self.assertRaises(ValueError):artifacts.import_tree(tree,target)

    def test_resume_when_server_ignores_range_and_reject_corruption(self):
        payload=b'recorded model data'*100
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);target=root/'output/weight.pt'
                target.parent.mkdir();target.with_suffix('.pt.partial').write_bytes(payload[:15])
                spec=dict(path='output/weight.pt',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                    url='http://127.0.0.1:%d/model'%server.server_port)
                download.download(spec,root)
                self.assertEqual(target.read_bytes(),payload)
                target.write_bytes(b'corrupted')
                with self.assertRaises(FileExistsError):download.download(spec,root)
        finally:
            server.shutdown();server.server_close();thread.join()

    def make_bundle(self,path,name='output/example.bin',digest=None):
        data=b'recorded artifact';sha=hashlib.sha256(data).hexdigest()
        manifest=json.dumps(dict(files=[dict(path=name,bytes=len(data),sha256=digest or sha)])).encode()
        with tarfile.open(path,'w') as archive:
            for n,b in [('artifact_manifest.json',manifest),(name,data)]:
                member=tarfile.TarInfo(n);member.size=len(b);archive.addfile(member,io.BytesIO(b))
        return data

    def test_import_is_verified_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=root/'data.tar';data=self.make_bundle(bundle)
            self.assertEqual(artifacts.import_bundle(bundle,root),1)
            self.assertEqual((root/'output/example.bin').read_bytes(),data)
            self.assertEqual(artifacts.import_bundle(bundle,root),1)
            self.assertEqual(artifacts.import_bundle(bundle,root,True),1)

    def test_traversal_and_corrupt_file_are_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bundle=root/'bad.tar'
            self.make_bundle(bundle,'output/../../escaped')
            with self.assertRaises(ValueError):artifacts.import_bundle(bundle,root)
            self.make_bundle(bundle,digest='0'*64)
            with self.assertRaises(ValueError):artifacts.import_bundle(bundle,root)
            self.assertFalse((root/'output/example.bin').exists())

    def test_pack_normalizes_paths_and_rejects_truncated_weights(self):
        import struct
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'output/local/weight.bin'
            source.parent.mkdir(parents=True);source.write_bytes(b'complete parameters')
            selection=root/'selection.json'
            selection.write_text(json.dumps([dict(source='output/local/weight.bin',path='output/model/weight.bin')]))
            bundle=root/'weights.tar'
            self.assertEqual(artifacts.pack_bundle(selection,bundle,root),1)
            with tarfile.open(bundle) as archive:
                self.assertNotIn(b'output/local',archive.extractfile('artifact_manifest.json').read())
            destination=root/'imported';destination.mkdir()
            artifacts.import_bundle(bundle,destination)
            self.assertEqual((destination/'output/model/weight.bin').read_bytes(),source.read_bytes())
            truncated=source.with_suffix('.safetensors')
            header=json.dumps(dict(weight=dict(dtype='F32',shape=[100],data_offsets=[0,400]))).encode()
            truncated.write_bytes(struct.pack('<Q',len(header))+header+b'partial')
            selection.write_text(json.dumps([dict(path='output/local/weight.safetensors')]))
            with self.assertRaises(ValueError):artifacts.pack_bundle(selection,root/'bad.tar',root)
            self.assertFalse((root/'bad.tar').exists())


if __name__=='__main__':unittest.main()
