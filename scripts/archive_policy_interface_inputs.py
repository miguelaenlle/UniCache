"""Freeze exact parity input records and relocated configs beside a run manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(out):
    manifest=json.loads((out/'manifest.json').read_text())
    root=out/'inputs'
    root.mkdir(exist_ok=True)
    snapshots=[]
    for case in manifest['cases']:
        original=Path(case['config'])
        config=yaml.safe_load(original.read_text())
        folder=root/case['name']
        folder.mkdir(exist_ok=True)
        (folder/'original.yaml').write_bytes(original.read_bytes())
        for i,task in enumerate(config['tasks']):
            source=Path(task['data_path'])
            if not source.is_absolute():source=Path(__file__).resolve().parents[1]/source
            destination=folder/f'data-{i}.jsonl'
            shutil.copyfile(source,destination)
            task['data_path']=str(destination.resolve())
            snapshots.append({'case':case['name'],'original':str(source),
                              'snapshot':str(destination.relative_to(out)),
                              'sha256':sha(destination)})
        (folder/'replay.yaml').write_text(yaml.safe_dump(config))
    (root/'snapshot.json').write_text(json.dumps(snapshots,indent=2))
    return snapshots


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    rows=archive(args.run.resolve())
    print(f'Archived {len(rows)} exact input files in {args.run}/inputs')

if __name__=='__main__':main()
