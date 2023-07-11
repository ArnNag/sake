eval "$(conda shell.bash hook)"
conda activate jax
bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=15] span[ptile=1]" -W 1:00 -n 1 -R A40 python eval.py _full_60_form_name_batch_32_eloss_0e+00_subset_3 3 3 
