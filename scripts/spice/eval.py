import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from functools import partial
import tqdm


def run(prefix):
    BATCH_SIZE = 800
    ds_tr = onp.load(prefix + "spice_train.npz")
    ds_vl = onp.load(prefix + "spice_valid.npz")

    # i_tr = ds_tr["atomic_numbers"]
    i_vl = ds_vl["atomic_numbers"]

    # Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
    ELEMENT_MAP = onp.array(
        [0, 1, -99, 2, -99, -99, 3, 4, 5, 6, -99, 7, 8, -99, -99, 9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99,
         -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99,
         -99, -99, -99, -99, 15])
    NUM_ELEMENTS = 16

    # i_tr = ELEMENT_MAP[i_tr]
    i_vl = ELEMENT_MAP[i_vl]
    # x_tr = ds_tr["pos"]
    x_vl = ds_vl["pos"]
    # f_tr = ds_tr["forces"]
    f_vl = ds_vl["forces"]
    y_tr = ds_tr["total_energy"]
    y_vl = ds_vl["total_energy"]
    y_tr = onp.expand_dims(y_tr, -1)
    y_vl = onp.expand_dims(y_vl, -1)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "f"]:
        for _split in ["vl"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])

    # i_tr = jax.nn.one_hot(i_tr, NUM_ELEMENTS)
    i_vl = jax.nn.one_hot(i_vl, NUM_ELEMENTS)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    class Model(nn.Module):
        def setup(self):
            self.model = sake.models.DenseSAKEModel(
                hidden_features=64,
                out_features=64,
                depth=6,
                update=False,
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

    def get_y_hat(params, i, x):
        m = make_edge_mask(i.argmax(-1) > 0)
        y_hat = model.apply(params, i, x, m=m)
        y_hat = coloring(y_hat)
        return y_hat

    from flax.training.checkpoints import restore_checkpoint
    for epoch in range(7):
        state = restore_checkpoint("_" + prefix, None, step=epoch)
        params = state['params']
        y_vl_hat_all = []
        num_batches = len(x_vl) // BATCH_SIZE
        for batch in range(num_batches):
            print(batch)
            x_vl_batch = x_vl[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            i_vl_batch = i_vl[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            y_vl_hat_all.append(get_y_hat(params, i_vl_batch, x_vl_batch))
        y_vl_hat = jnp.concatenate(y_vl_hat_all)
        print("epoch: ", epoch, "validation:", sake.utils.bootstrap_mae(y_vl_hat, y_vl[:num_batches * BATCH_SIZE]))


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
