eval "$(conda shell.bash hook)"
conda activate jax
bsub -q gpuqueue -gpu "num=1:j_exclusive=yes" -o %J.stdout -R "rusage[mem=10] span[ptile=1]" -R A40 -W 1:00 -n 1 python sparse.py "full_96_dist_" 32 0 3


