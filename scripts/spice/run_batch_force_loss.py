import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run():
    ds_tr, ds_vl, ds_te = onp.load("spice_train.npz"), onp.load("spice_val.npz"), onp.load("spice_test.npz")
    i_tr, i_vl, i_te = ds_tr["atomic_numbers"], ds_vl["atomic_numbers"], ds_te["atomic_numbers"]
    x_tr, x_vl, x_te = ds_tr["pos"], ds_vl["pos"], ds_te["pos"]
    f_tr, f_vl, f_te = ds_tr["forces"], ds_vl["forces"], ds_te["forces"]
    y_tr, y_vl, y_te = ds_tr["total_energy"], ds_vl["total_energy"], ds_te["total_energy"]
    
    y_tr, y_vl, y_te = onp.expand_dims(y_tr, -1), onp.expand_dims(y_vl, -1), onp.expand_dims(y_te, -1)
    m_tr, m_vl, m_te = (i_tr > 0), (i_vl > 0), (i_te > 0)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "f", "m"]:
        for _split in ["tr", "vl", "te"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])


    i_tr, i_vl, i_te = jax.nn.one_hot(i_tr, i_tr.max()+1), jax.nn.one_hot(i_vl, i_tr.max()+1), jax.nn.one_hot(i_te, i_tr.max()+1)
    m_tr, m_vl, m_te = make_edge_mask(m_tr), make_edge_mask(m_vl), make_edge_mask(m_te)

    BATCH_SIZE = 64
    N_BATCHES = len(i_tr) // BATCH_SIZE

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    print(y_tr.mean(), y_tr.std())

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
        e_pred = get_e_pred(params, i, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, x: jax.grad(get_e_pred_sum, argnums=(2,))(params, x)[0])

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
    def epoch(state, i_tr, x_tr, m_tr, y_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(BATCH_SIZE * N_BATCHES))
        _i_tr = i_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *i_tr.shape[1:])
        _x_tr = x_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *x_tr.shape[1:])
        _m_tr = m_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *m_tr.shape[1:])
        _y_tr = y_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, 1)

        def loop_body(idx, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            i, x, m, y = _i_tr[idx], _x_tr[idx], _m_tr[idx], _y_tr[idx]
            print("i.shape", i.shape, "x.shape", x.shape) 
            state = step_with_loss(state, i, x, m, y)
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)

        '''
        for i, x, m, y in iterator: 
            state = loop_body(i, x, m, y, state)
        '''

        return state

    @partial(jax.jit, static_argnums=(5))
    def many_epochs(state, i_tr, x_tr, m_tr, y_tr, n=10):
        def loop_body(idx_batch, state):
            state = epoch(state, i_tr, x_tr, m_tr, y_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    i0 = i_tr[:BATCH_SIZE]
    x0 = x_tr[:BATCH_SIZE]
    m0 = m_tr[:BATCH_SIZE]
    y0 = y_tr[:BATCH_SIZE]

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

    for idx_batch in tqdm.tqdm(range(2000)):
        state = epoch(state, i_tr, x_tr, m_tr, y_tr)
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint("_" + target, target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run()
