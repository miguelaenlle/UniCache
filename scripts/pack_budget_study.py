"""Package reproduction inputs and reports, excluding caches and raw replay logs."""
import argparse
import gzip
import hashlib
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study=a.study.resolve();output=a.output.resolve()
    if output.exists():raise ValueError('Archive already exists')
    allowed={'.json','.csv','.md','.png','.txt','.sh'}
    files={f for f in study.iterdir() if f.is_file() and f.suffix in allowed}
    files|={f for f in (study/'inputs').rglob('*') if f.is_file()}
    for folder in study.iterdir():
        if folder.is_dir() and (folder.name in ('pilot','reference') or folder.name.startswith('confirm-')):
            files|={f for f in folder.iterdir() if f.is_file() and f.suffix in allowed}
            if (folder/'source').exists():files|={f for f in (folder/'source').rglob('*') if f.is_file()}
    files=sorted(f for f in files if not f.is_symlink())
    checks=[]
    for f in files:
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        checks.append(h.hexdigest()+'  '+str(f.relative_to(study)))
    checksum=study/'SHA256SUMS';checksum.write_text('\n'.join(checks)+'\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as archive:
        for f in [*files,checksum]:
            item=archive.gettarinfo(str(f),arcname=str(Path(study.name)/f.relative_to(study)))
            item.mtime=0;item.uid=item.gid=0;item.uname=item.gname=''
            with f.open('rb') as stream:archive.addfile(item,stream)
    print(output);print(f'{len(files)} files; {output.stat().st_size/2**20:.1f} MiB compressed')

if __name__=='__main__':main()
