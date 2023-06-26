eval "$(conda shell.bash hook)"
conda activate sake
bsub -q cpuqueue -o %J.stdout -R "rusage[mem=10] span[ptile=1]" -W 03:00 -n 1 python spiceloadernp.py full_60_ 0.8 0.1 60
