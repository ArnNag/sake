import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run(prefix):
 
    ds_tr, ds_vl = onp.load(prefix + "spice_train.npz"), onp.load(prefix + "spice_valid.npz")

    i_tr, i_vl = ds_tr["atomic_numbers"], ds_vl["atomic_numbers"]
    # Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
    ELEMENT_MAP = onp.array([ 0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15]) 
    NUM_ELEMENTS = 16
    i_tr, i_vl = ELEMENT_MAP[i_tr], ELEMENT_MAP[i_vl]

    x_tr, x_vl = ds_tr["pos"], ds_vl["pos"]
    f_tr, f_vl = ds_tr["forces"], ds_vl["forces"]
    y_tr, y_vl = ds_tr["total_energy"], ds_vl["total_energy"]
    
    y_tr, y_vl = onp.expand_dims(y_tr, -1), onp.expand_dims(y_vl, -1)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "m", "f"]:
        for _split in ["tr", "vl"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])


    i_tr, i_vl = jax.nn.one_hot(i_tr, NUM_ELEMENTS), jax.nn.one_hot(i_vl, NUM_ELEMENTS)

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

    def get_y_hat(params, i, x, m):
        y_hat = model.apply(params, i, x, m=m)
        y_hat = coloring(y_hat)
        return y_hat

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_" + prefix, None)
    params = state['params']

    def _get_y_hat(inputs):
         x, i = inputs
         m = make_edge_mask(i.argmax(-1) > 0)
         return get_y_hat(params, i, x, m)
    
    y_tr_hat = jax.lax.map(_get_y_hat, (x_tr, i_tr))

    y_vl_hat = jax.lax.map(_get_y_hat, (x_vl, i_vl))

    print(y_tr_hat)
    
    print("training", sake.utils.bootstrap_mae(y_tr_hat, y_tr))
    print("validation", sake.utils.bootstrap_mae(y_vl_hat, y_vl))

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
