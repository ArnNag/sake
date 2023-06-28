import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from jax_tqdm import loop_tqdm

def run(prefix):
    ds_tr = onp.load(prefix + "spice_train.npz")
    i_tr = ds_tr["atomic_numbers"]

    # Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
    ELEMENT_MAP = onp.array([ 0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15]) 
    NUM_ELEMENTS = 16
    i_tr = ELEMENT_MAP[i_tr]

    x_tr = ds_tr["pos"]
    f_tr = ds_tr["forces"]
    y_tr = ds_tr["total_energy"]
    
    print("loaded all data")

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "f"]:
        for _split in ["tr"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])

    BATCH_SIZE = 32
    N_BATCHES = len(i_tr) // BATCH_SIZE

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    class Model(nn.Module):
        def setup(self):
            self.model = sake.models.DenseSAKEModel(
                hidden_features=64,
                out_features=64,
                depth=6,
                update=False, # [False, False, False, True, True, True],
            )

            self.mlp = nn.Sequential(
                [
                    nn.Dense(64),
                    nn.silu,
                    nn.Dense(1),
                ],
            )

        def __call__(self, i, x, m):
            y, _, __ = self.model(i, x, mask=m)
            y = y * sum_mask(m)
            y = y.sum(-2)
            y = self.mlp(y)
            return y

    model = Model()


    @jax.jit
    def get_e_pred(params, i, x, m):
        e_pred = model.apply(params, i, x, m)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x, m):
        e_pred = get_e_pred(params, i, x, m)
        return -e_pred.sum()

    get_f_pred = jax.jit(jax.grad(get_e_pred_sum, argnums=2))

    def loss_fn(params, i, x, m, f, y):
        e_pred = get_e_pred(params, i, x, m)
        f_pred = get_f_pred(params, i, x, m)
        e_loss = jnp.abs(e_pred - y).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step_with_loss(state, i, x, m, f, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, f, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def epoch(state, i_tr, x_tr, f_tr, y_tr):
        loader = SPICEBatchLoader(i_tr, x_tr, f_tr, y_tr, state.step, BATCH_SIZE, NUM_ELEMENTS)

        @loop_tqdm(N_BATCHES)
        def loop_body(idx, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            i, x, m, f, y = loader.get_batch(idx)  
            state = step_with_loss(state, i, x, m, f, y)
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)

        return state

    @partial(jax.jit, static_argnums=(4))
    def many_epochs(state, i_tr, x_tr, y_tr, n=10):
        def loop_body(idx_batch, state):
            state = epoch(state, i_tr, x_tr, y_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    init_loader = SPICEBatchLoader(i_tr, x_tr, f_tr, y_tr, 2666, BATCH_SIZE, NUM_ELEMENTS)
    i0, x0, m0, _, __ = init_loader.get_batch(0)

    key = jax.random.PRNGKey(2666)
    params = model.init(key, i0, x0, m0)

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=100 * N_BATCHES,
        decay_steps=1900 * N_BATCHES,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )

    optimizer = optax.apply_if_finite(optimizer, 5)


    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )


    for idx_batch in range(7):
        print("before epoch")
        state = epoch(state, i_tr, x_tr, f_tr, y_tr)
        print("after epoch")
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint("_" + prefix, target=state, keep=7, step=idx_batch)

'''
Initialize for every epoch with a unique seed.
'''
class SPICEBatchLoader:

    def __init__(self, i_tr, x_tr, f_tr, y_tr, seed, batch_size, num_elements):
        self.batch_size = batch_size
        self.num_elements = num_elements
        self.i_tr = i_tr
        self.x_tr = x_tr
        self.f_tr = f_tr
        self.y_tr = y_tr
        key = jax.random.PRNGKey(seed)
        n_batches = len(i_tr) // batch_size
        self.idxs = jax.random.permutation(key, n_batches * batch_size).reshape(n_batches, batch_size)

    def get_batch(self, batch_num):
        batch_idxs = self.idxs[batch_num]
        i_nums = self.i_tr[batch_idxs]
        i_batch = jax.nn.one_hot(i_nums, self.num_elements) 
        x_batch = self.x_tr[batch_idxs]
        f_batch = self.f_tr[batch_idxs]
        _m = i_nums > 0
        m_batch = jnp.einsum("bn,bN->bnN", _m, _m) 
        y_batch = jnp.expand_dims(self.y_tr[batch_idxs], -1)
        return i_batch, x_batch, m_batch, f_batch, y_batch  

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
