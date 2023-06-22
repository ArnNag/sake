eval "$(conda shell.bash hook)"
conda activate sake
bsub -q gpuqueue -o %J.stdout -R "rusage[mem=20] span[ptile=1]" -W 00:01 -n 1 python run_batch_force_loss.py
