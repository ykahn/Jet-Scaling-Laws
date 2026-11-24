#!/bin/bash
#SBATCH --account=aip-yfkahn
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
source ~/jsl-env/bin/activate
python - "$@" <<'EOF'
import json, sys
name, keep = sys.argv[1], sys.argv[2:]     # notebook name, then headings of the sections to run (none = all)
sec, out = '', []
for c in json.load(open(name + '.ipynb'))['cells']:
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        if src.lstrip().startswith('#'): sec = src.split('\n')[0]
    elif not keep or not out or any(k in sec for k in keep):     # first code cell (imports) always runs
        out.append(src)
open(name + '.py', 'w').write('\n\n'.join(out) + '\n')
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU"
python -u "$1.py"
