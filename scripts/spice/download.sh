bsub -q cpuqueue -o %J.stdout -W 10:00 -n 1 wget https://zenodo.org/record/7606550/files/SPICE-1.1.3.hdf5
