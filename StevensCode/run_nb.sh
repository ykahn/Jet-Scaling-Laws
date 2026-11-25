#!/bin/bash
#SBATCH --account=aip-yfkahn
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
source ~/jsl-env/bin/activate
script="${SLURM_TMPDIR:-/tmp}/$1.${SLURM_JOB_ID:-local}.py"     # private copy for this job, so parallel jobs can't collide
python - "$script" "$@" <<'EOF'
import json, sys
script, name, keep = sys.argv[1], sys.argv[2], sys.argv[3:]     # output file, notebook name, headings of the sections to run (none = all)
sec, out = '', []
for c in json.load(open(name + '.ipynb'))['cells']:
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        if src.lstrip().startswith('#'): sec = src.split('\n')[0]
    elif not keep or not out or any(k in sec for k in keep):     # first code cell (imports) always runs
        out.append(src)
open(script, 'w').write('\n\n'.join(out) + '\n')
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU"
PYTHONPATH="$PWD:$PYTHONPATH" python -u "$script"
