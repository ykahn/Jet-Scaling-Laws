#!/bin/bash
# Submit one job per sample. Run as:  bash submit.sh
read -p "Notebook (e.g. Measure_r): " nb
read -p "Samples (space-separated): " -a keys
read -p "Sections (comma-separated, blank = all cells): " secs
read -p "Time limit per job [00:45:00]: " t
IFS=',' read -ra sec <<< "$secs"
for k in "${keys[@]}"; do KEYS=$k sbatch --gres=gpu:l40s:1 --time="${t:-00:45:00}" run_nb.sh "$nb" "${sec[@]}"; done
