eval "$(conda shell.bash hook)"
conda activate sake
bsub -q cpuqueue -o %J.stdout -R "rusage[mem=100] span[ptile=1]" -W 24:00 -n 1 python spiceserializer.py full_96_dist_ 0.8 0.1 96
