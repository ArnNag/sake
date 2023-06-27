eval "$(conda shell.bash hook)"
conda activate jax
bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -o %J.stdout -R "rusage[mem=5] span[ptile=1]" -R A40 -W 24:00 -n 1 python run.py "full_60_"


