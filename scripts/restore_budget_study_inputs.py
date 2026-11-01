"""Verify archived study inputs; optionally restore missing files in a fresh checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest_prefix(path,lines=None):
    h=hashlib.sha256()
    with path.open('rb') as f:
        if lines is not None:
            for _ in range(lines):
                row=f.readline()
                if not row:raise ValueError('Source has fewer sessions than required')
                h.update(row)
        else:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--restore-missing',action='store_true',help='Copy archived inputs only if destination files do not exist')
    a=p.parse_args();study=a.study.resolve();repo=a.repo.resolve()
    info=json.loads((study/'inputs.json').read_text())
    for trace,item in info['prepared_sources'].items():
        source=(study/item['file']).resolve();target=(repo/item['restore_to']).resolve()
        if not source.is_relative_to(study) or not target.is_relative_to(repo):raise ValueError('Unsafe input path')
        if digest_prefix(source)!=item['sha256']:raise ValueError('Archive checksum mismatch: '+trace)
        if not target.exists() and a.restore_missing:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if not target.exists():raise ValueError('Input missing; use --restore-missing: '+str(target))
        if digest_prefix(target,item['sessions'])!=item['sha256']:raise ValueError('Prepared prefix mismatch: '+trace)
        print(trace+': verified')
    for source in (study/'inputs/model-metadata').iterdir():
        target=repo/'models/Llama-3.2-1B-Instruct'/source.name
        if not target.exists() and a.restore_missing:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if not target.exists() or digest_prefix(target)!=digest_prefix(source):raise ValueError('Model metadata mismatch: '+source.name)
    print('All required prepared inputs and model metadata verified. No existing files overwritten.')

if __name__=='__main__':main()
