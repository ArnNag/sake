import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import flax
import numpy as onp
import sake
import tqdm
from functools import partial
from utils import ELEMENT_MAP, NUM_ELEMENTS, select, SPICEBatchLoader, get_e_pred, get_f_pred, SparseSAKEEnergyModel, loss_fn
from functools import partial

def run(prefix, max_nodes=3600, max_edges=60000, max_graphs=152, e_loss_factor=0, subset=None):
    ds_tr = onp.load(prefix + "spice_train.npz")
    i_tr = ELEMENT_MAP[ds_tr["atomic_numbers"]]
    x_tr = ds_tr["pos"]
    f_tr = ds_tr["forces"]
    y_tr = ds_tr["formation_energy"]
    edges_tr = ds_tr["edges"]
    num_nodes_tr = ds_tr["num_nodes"]
    num_edges_tr = ds_tr["num_edges"]
    subset_labels = ds_tr["subsets"]

    if subset >= 0: 
        i_tr, x_tr, f_tr, y_tr, edges_tr = select(subset_labels, subset, i_tr, x_tr, f_tr, y_tr, edges_tr)
    
    print("loaded all data")

    for _var in ["i", "x", "y", "f"]:
        for _split in ["tr"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])



    model = SparseSAKEEnergyModel(num_segments=max_graphs)

    @partial(jax.jit, static_argnums=(0,))
    def step_with_loss(model, state, i, x, edges, f, y, graph_segments):
        variables = state.params
        grads = jax.grad(loss_fn, argnums=1)(model, variables, i, x, edges, f, y, graph_segments, e_loss_factor)
        state = state.apply_gradients(grads=grads)
        return state
    
    def epoch(model, state, i_tr, x_tr, edges_tr, f_tr, y_tr):
        loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=state.step, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs, num_elements=NUM_ELEMENTS)

        for idx in tqdm.tqdm(range(len(loader))):
            i, x, edges, f, y, graph_segments = loader.get_batch(idx)  
            state = step_with_loss(model, state, i, x, edges, f, y, graph_segments)

        return state

    init_loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=2666, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs, num_elements=NUM_ELEMENTS)
    i0, x0, edges0, _, __, graph_segments0 = init_loader.get_batch(0)

    key = jax.random.PRNGKey(2666)
    variables = model.init(key, i0, x0, edges0, graph_segments0)
    variables = variables.copy({'coloring': {'mean': y_tr.mean(), 'std': y_tr.std()}})

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=100 * len(i_tr) // len(init_loader),
        decay_steps=1900 * len(i_tr) // len(init_loader),
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )

    optimizer = optax.apply_if_finite(optimizer, 5)

    partition_optimizers = {'trainable': optimizer, 'frozen': optax.set_to_zero()}
    param_partitions = flax.core.freeze(flax.traverse_util.path_aware_map(
      lambda path, v: 'frozen' if 'coloring' in path else 'trainable', variables))
    masked_optimizer = optax.multi_transform(partition_optimizers, param_partitions)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=variables, tx=masked_optimizer
    )


    NUM_EPOCHS = 100
    for epoch_num in range(NUM_EPOCHS):
        print("before epoch")
        state = epoch(model, state, i_tr, x_tr, edges_tr, f_tr, y_tr)
        print("after epoch")
        print("state.opt_state.notfinite_count:", state.opt_state.notfinite_count)
        # assert state.opt_state.notfinite_count <= 10
        save_checkpoint(f"_sparse_{prefix}eloss_{e_loss_factor:.0e}_subset_{subset}", target=state, keep_every_n_steps=10, step=epoch_num)


if __name__ == "__main__":
    import sys
    run(sys.argv[1], e_loss_factor=float(sys.argv[2]), subset=int(sys.argv[3]))
