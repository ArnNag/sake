eval "$(conda shell.bash hook)"
conda activate jax
bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=15] span[ptile=1]" -W 0:10 -n 1 python eval.py _full_60_form_name_batch_56_eloss_1.000000e+00_subset_3 3 3
