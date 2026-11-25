#!/usr/bin/env python3
"""Pack everything DeltaAI needs into result/sim2real-transfer/bundle/.

  unicache-src.tar.gz  tracked code at HEAD + Llama-3.2-1B config/tokenizer (no weights)
  subsets/<w>/         subset inputs (eval.jsonl, eval.yaml) for the given workloads
  agents/<base>.py     the agent policies, with agents/SHA256SUMS
  *.def, *.sbatch      container recipe and job scripts

    .venv-sim312/bin/python scripts/sim2real/make_bundle.py alfred codeparrot@600 sharegpt@100x2 codeparrot@120
"""
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, RESULTS, ROOT, SUBSETS, agent_policy, spec  # noqa: E402

out = RESULTS / "bundle"
shutil.rmtree(out, ignore_errors=True)
(out / "agents").mkdir(parents=True)
if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                  capture_output=True, text=True).stdout.strip():
    sys.exit("Commit first: the bundle is built from HEAD")
src = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT,
                     capture_output=True, check=True).stdout
with tarfile.open(out / "unicache-src.tar.gz", "w:gz") as tar:
    with tarfile.open(fileobj=io.BytesIO(src)) as code:
        for m in code.getmembers():
            tar.addfile(m, code.extractfile(m) if m.isfile() else None)
    for f in MODEL.iterdir():
        if f.is_file():
            tar.add(f, arcname=f"models/{MODEL.name}/{f.name}")
sums = []
for w in sys.argv[1:]:
    shutil.copytree(SUBSETS / w, out / "subsets" / w)
    base = spec(w)["base"]
    dst = out / "agents" / f"{base}.py"
    if not dst.exists():
        shutil.copy(agent_policy(w), dst)
        sums.append(f"{hashlib.sha256(dst.read_bytes()).hexdigest()}  {base}.py  <- {agent_policy(w)}")
shutil.copy(SUBSETS / "subsets.json", out / "subsets" / "subsets.json")
(out / "agents" / "SHA256SUMS").write_text("\n".join(sums) + "\n")
for f in (Path(__file__).parent / "delta").iterdir():
    shutil.copy(f, out / f.name)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
(out / "COMMIT").write_text(head)
print(subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.strip())
