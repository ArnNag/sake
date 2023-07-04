eval "$(conda shell.bash hook)"
conda activate jax
bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=50] span[ptile=1]" -W 3:59 -n 1 -R A40 python eval.py full_60_
