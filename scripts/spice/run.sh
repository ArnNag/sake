eval "$(conda shell.bash hook)"
conda activate sake
bsub -u "lt-gpu" -q gpuqueue -gpu "num=1:j_exclusive=yes" -o %J.stdout -R "rusage[mem=100] span[ptile=1]" -W 10:00 -n 1 python run_batch_force_loss.py "small_"
