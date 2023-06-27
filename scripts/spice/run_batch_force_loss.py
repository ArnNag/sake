import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import config
import numpy as onp
import sake
import tqdm

def run(prefix):
    ds_tr, ds_vl, ds_te = onp.load(prefix + "spice_train.npz"), onp.load(prefix + "spice_val.npz"), onp.load(prefix + "spice_test.npz")
    i_tr, i_vl, i_te = ds_tr["atomic_numbers"], ds_vl["atomic_numbers"], ds_te["atomic_numbers"]

    # Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
    ELEMENT_MAP = onp.array([ 0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15]) 
    NUM_ELEMENTS = 16
    i_tr, i_vl, i_te = ELEMENT_MAP[i_tr], ELEMENT_MAP[i_vl], ELEMENT_MAP[i_te]

    x_tr, x_vl, x_te = ds_tr["pos"], ds_vl["pos"], ds_te["pos"]
    f_tr, f_vl, f_te = ds_tr["forces"], ds_vl["forces"], ds_te["forces"]
    y_tr, y_vl, y_te = ds_tr["total_energy"], ds_vl["total_energy"], ds_te["total_energy"]
    
    print("loaded all data")

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "f"]:
        for _split in ["tr", "vl", "te"]:
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
        i_tr = jnp.broadcast_to(i, (*x.shape[:-1], i.shape[-1]))
        e_pred = model.apply(params, i_tr, x, m)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x, m):
        e_pred = get_e_pred(params, i, x, m)
        return -e_pred.sum()

    get_f_pred = jax.jit(jax.grad(get_e_pred_sum, argnums=(2)))

    def loss_fn(params, i, x, m, f, y):
        e_pred = get_e_pred(params, i, x, m)
        f_pred = get_f_pred(params, i, x, m)
        e_loss = jnp.abs(e_pred - y).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step_with_loss(state, i, x, m, f, y):
        params = state.params
        print("before grads", i)
        grads = jax.grad(loss_fn)(params, i, x, m, f, y)
        print("after grads")
        state = state.apply_gradients(grads=grads)
        print("after apply")
        return state
    
    @jax.jit
    def epoch(state, i_tr, x_tr, f_tr, y_tr):
        print("start of epoch")
        loader = SPICEBatchLoader(i_tr, x_tr, f_tr, y_tr, state.step, BATCH_SIZE, NUM_ELEMENTS)

        def loop_body(idx, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            i, x, m, f, y = loader.get_batch(idx)  
            state = step_with_loss(state, i, x, m, f, y)
            print("after step_with_loss")
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)
        print("after fori_loop")

        return state

    @partial(jax.jit, static_argnums=(4))
    def many_epochs(state, i_tr, x_tr, y_tr, n=10):
        def loop_body(idx_batch, state):
            state = epoch(state, i_tr, x_tr, y_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    init_loader = SPICEBatchLoader(i_tr, x_tr, f_tr, y_tr, 2666, BATCH_SIZE, NUM_ELEMENTS)
    i0, x0, m0, _, _ = init_loader.get_batch(0)

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


    for idx_batch in range(5):
        print("before epoch")
        state = epoch(state, i_tr, x_tr, f_tr, y_tr)
        print("after epoch")
        assert state.opt_state.notfinite_count <= 10
        # save_checkpoint("_" + target, target=state, step=idx_batch)

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
        self.idxs = jax.random.permutation(key, batch_size * n_batches)

    def get_batch(self, batch_num):
        batch_start = batch_num * self.batch_size
        batch_end = batch_start + self.batch_size
        batch_idxs = self.idxs[batch_start:batch_end]
        i_nums = self.i_tr[batch_idxs]
        i_batch = jax.nn.one_hot(i_nums, self.num_elements) 
        x_batch = self.x_tr[batch_idxs]
        f_batch = self.f_tr[batch_idxs]
        _m = i_nums > 0
        m_batch = jnp.einsum("bn,bN->bnN", _m, _m) 
        y_batch = onp.expand_dims(self.y_tr[batch_idxs], -1)
        jax.debug.print("i_nums: {i_nums}, i_batch: {i_batch}, x_batch: {x_batch}, f_batch: {f_batch}, _m: {_m}, m_batch: {m_batch}, y_batch: {y_batch}", i_nums=i_nums.shape, i_batch=i_batch.shape, x_batch=x_batch.shape, f_batch=f_batch.shape, _m=_m.shape, m_batch=m_batch.shape, y_batch=y_batch.shape)
        return i_batch, x_batch, f_batch, m_batch, y_batch  

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
