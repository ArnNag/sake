import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
from spiceloadernp import SpiceLoader

def run():

    loader = SpiceLoader("spice_train")  

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    BATCH_SIZE = 64
    N_BATCHES = len(loader) // BATCH_SIZE

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=0, std=1) # TODO: serialize mean and std

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
            print("i shape: ", i.shape, "\nx shape", x.shape, "\nm shape", m.shape)
            y, _, __ = self.model(i, x, mask=m)
            y = y * sum_mask(m)
            y = y.sum(-2)
            y = self.mlp(y)
            return y

    model = Model()

    def get_y_hat(params, i, x, m):
        y_hat = model.apply(params, i, x, m=m)
        y_hat = coloring(y_hat)
        return y_hat

    def loss_fn(params, i, x, m, y):
        y_hat = get_y_hat(params, i, x, m)
        loss = jnp.abs(y - y_hat).mean()
        return loss

    def step(state, i, x, m, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def step_with_loss(state, i, x, m, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def epoch(state, loader):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(BATCH_SIZE * N_BATCHES))

        def loop_body(batch_num, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            batch_start = BATCH_SIZE * batch_num
            batch_end = batch_start + BATCH_SIZE
            batch_idxs = idxs[batch_start:batch_end]
            _i, x, y = loader[batch_idxs]
            i = jax.nn.one_hot(i, i.max()+1)
            m = make_edge_mask(i > 0)
            state = step_with_loss(state, i, x, m, y)
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)

        '''
        for i, x, m, y in iterator: 
            state = loop_body(i, x, m, y, state)
        '''

        return state

    @partial(jax.jit, static_argnums=(3))
    def many_epochs(state, loader, n=10):
        def loop_body(idx_batch, state):
            state = epoch(state, loader)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    i0, x0, y0 = loader[:BATCH_SIZE]
    m0 = make_edge_mask(i0 > 0)

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
        save_checkpoint("_spice_batch", target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run()
