eval "$(conda shell.bash hook)"
conda activate sake
bsub -q cpuqueue -o %J.stdout -R "rusage[mem=50] span[ptile=1]" -W 24:00 -n 1 python spiceserializer.py small_96_dist_nums_ 0.005 0 96
