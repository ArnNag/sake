import jax
import optax
import flax
from utils import load_data, make_batch_loader, SAKEEnergyModel, get_y_loss, get_f_loss
from functools import partial


def run(prefix, max_nodes=3600, max_edges=60000, max_graphs=152, e_loss_factor=0., subset=-1):
    graph_list, y_mean, y_std = load_data(prefix + "spice_train.npz", subset)
    print("loaded all data")

    model = SAKEEnergyModel()

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(model, params, graph):
        e_loss = get_y_loss(model, params, graph)
        f_loss = get_f_loss(model, params, graph)
        return f_loss + e_loss * e_loss_factor

    @partial(jax.jit, static_argnums=(0,))
    def step_with_loss(model, state, graph):
        variables = state.params
        grads = jax.grad(loss_fn, argnums=1)(model, variables, graph)
        state = state.apply_gradients(grads=grads)
        return state
    
    def epoch(model, state, graph_list):
        loader = make_batch_loader(graph_list, seed=state.step, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs)

        for graph in loader:
            state = step_with_loss(model, state, graph)

        return state

    init_loader = list(make_batch_loader(graph_list, seed=0, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs))
    init_graph = init_loader[0]

    key = jax.random.PRNGKey(2666)
    variables = model.init(key, init_graph)
    variables = variables.copy({'coloring': {'mean': y_mean, 'std': y_std}})

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=100 * len(graph_list) // len(init_loader),
        decay_steps=1900 * len(graph_list) // len(init_loader),
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
        state = epoch(model, state, graph_list)
        print("after epoch")
        print("notfinite_count:", state.opt_state.inner_states["trainable"].inner_state.notfinite_count)
        # assert state.opt_state.notfinite_count <= 10
        save_checkpoint(f"_sparse_{prefix}eloss_{e_loss_factor:.0e}_subset_{subset}", target=state, keep_every_n_steps=10, step=epoch_num)


if __name__ == "__main__":
    import sys
    run(sys.argv[1], e_loss_factor=float(sys.argv[2]), subset=int(sys.argv[3]))
