eval "$(conda shell.bash hook)"
conda activate jax
bsub -q cpuqueue -o %J.stdout -R "rusage[mem=15] span[ptile=1]" -W 1:00 -n 1 python eval_sparse.py _sparse_full_96_dist_nums_eloss_0e+00_subset_3 small_96_dist_nums_spice_train.npz 
