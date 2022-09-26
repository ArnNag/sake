pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tensorflow tqdm
export PYTHONPATH=$PYTHONPATH:~/sake/

learninte_rate = 1e-4
weight_decay = 1e-10
batch_size = 32

python3 run.py \
    --target $target \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --batch_size $batch_size

