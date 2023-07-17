import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from jax_tqdm import loop_tqdm
from utils import ELEMENT_MAP, NUM_ELEMENTS, select

def run(prefix, batch_size=32, e_loss_factor=1, subset=None):
    ds_tr = onp.load(prefix + "spice_train.npz")
    i_tr = ELEMENT_MAP[ds_tr["atomic_numbers"]]
    x_tr = ds_tr["pos"]
    f_tr = ds_tr["forces"]
    y_tr = ds_tr["formation_energy"]
    edges_tr = ds_tr["edges"]
    subset_labels = ds_tr["subsets"]

    if subset >= 0: 
        i_tr, x_tr, f_tr, y_tr, edges_tr = select(subset_labels, subset, i_tr, x_tr, f_tr, y_tr, edges_tr)
    
    print("loaded all data")

    for _var in ["i", "x", "y", "f"]:
        for _split in ["tr"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])

    N_BATCHES = len(i_tr) // batch_size

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    class Model(nn.Module):
        def setup(self):
            self.model = sake.models.SparseSAKEModel(
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

        def __call__(self, i, x, edges):
            y = self.model(i, x, edges=edges)[0]
            y = y.sum(-2)
            y = self.mlp(y)
            return y

    model = Model()


    @jax.jit
    def get_e_pred(params, i, x, edges):
        e_pred = model.apply(params, i, x, edges)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x, edges):
        e_pred = get_e_pred(params, i, x, edges)
        return -e_pred.sum()

    get_f_pred = jax.jit(jax.grad(get_e_pred_sum, argnums=2))

    def loss_fn(params, i, x, edges, f, y):
        e_pred = get_e_pred(params, i, x, edges)
        f_pred = get_f_pred(params, i, x, edges)
        e_loss = jnp.abs(e_pred - y).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * e_loss_factor

    @jax.jit
    def step_with_loss(state, i, x, edges, f, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, edges, f, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def epoch(state, i_tr, x_tr, edges_tr, f_tr, y_tr):
        loader = SPICEBatchLoader(i_tr, x_tr, edges_tr, f_tr, y_tr, state.step, batch_size, NUM_ELEMENTS)

        @loop_tqdm(N_BATCHES)
        def loop_body(idx, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            i, x, edges, f, y, graph_segments = loader.get_batch(idx)  
            state = step_with_loss(state, i, x, edges, f, y)
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

    init_loader = SPICEBatchLoader(i_tr, x_tr, edges_tr, f_tr, y_tr, 2666, batch_size, NUM_ELEMENTS)
    i0, x0, m0 = init_loader.get_batch(0)[:2]

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


    NUM_EPOCHS = 100
    for idx_batch in range(NUM_EPOCHS):
        print("before epoch")
        state = epoch(state, i_tr, x_tr, edges_tr, f_tr, y_tr)
        print("after epoch")
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint(f"_{prefix}batch_{batch_size}_eloss_{e_loss_factor:.0e}_subset_{subset}", target=state, keep_every_n_steps=10, step=idx_batch)

'''
Initialize for every epoch with a unique seed.
'''
class SPICEBatchLoader:

    def __init__(self, i_tr, x_tr, edges_tr, f_tr, y_tr, num_nodes_tr, num_edges_tr, seed, max_edges, max_nodes, max_graphs, num_elements):
        self.max_graphs = max_graphs
        self.num_elements = num_elements
        self.i_tr = i_tr
        self.x_tr = x_tr
        self.edges_tr = edges_tr
        self.f_tr = f_tr
        self.y_tr = y_tr
        self.num_nodes_tr = num_nodes_tr
        self.num_edges_tr = num_edges_tr
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_graphs = max_graphs
        key = jax.random.PRNGKey(seed)
        self.idxs = jax.random.permutation(key, len(i_tr))
        self.batch_list, self.graph_segments = _make_batch_list()

    def _make_batch_list():
        total_graphs_added = 0
        batch_list = []
        graph_segment_list = []
        while total_graphs_added < len(self.idxs):
            batch_idxs = []
            batch_graph_segments = []
            batch_nodes_added = 0
            batch_edges_added = 0
            while True:
                tr_idx = self.idxs[total_graphs_added]
                batch_nodes_added += self.num_nodes_tr[tr_idx] 
                batch_edges_added += self.num_edges_tr[tr_idx]
                if len(batch_idxs) >= self.max_graphs or batch_nodes_added >= self.max_nodes or batch_edges_added >= self.max_edges:
                    break
                batch_graph_segments.extend([len(batch_idxs)] * self.num_nodes_tr[tr_idx])
                batch_idxs.append(tr_idx)
                total_graphs_added += 1
            batch_list.append(batch_idxs)
            graph_segment_list.append(batch_graph_segments.extend([-1] * (max_graphs - len(batch_idxs))))
        return batch_list, jnp.array(graph_segment_list)


    def get_batch(self, batch_num):
        batch_idxs = self.batch_list[batch_num]
        batch_graph_segments = self.graph_segments[batch_num]
        batch_num_nodes = self.num_nodes_tr[batch_idxs]
        batch_num_edges = self.num_edges_tr[batch_idxs]

        def flatten_data(batch_data, batch_num_data, max_num_data, fill_value, offsets)
            flattened_data = jnp.fill((max_num_data, *data.shape[2:]), fill_value)
            data_flattened = 0
            for i, (graph_data, graph_num_data) in enumerate(zip(batch_data, batch_num_data)):
                next_data_idx = data_flattened + graph_num_data
                data_fill = graph_data[:graph_num_data]
                if offsets is not None:
                    data_fill += offsets[i]
                flattened_data[data_flattened:next_data_idx] = graph_data[:graph_num_data]

                data_flattened = next_data_idx
            return flattened_data

        def flatten_nodes(batch_data):
            return flatten_data(batch_data, batch_num_nodes, self.max_nodes, 0, None)

        def flatten_edges(batch_data):
            return flatten_data(batch_data, batch_num_edges, self.max_edges, -1, jnp.cumsum([0] + batch_num_nodes[:-1]))

        i_nums = flatten_nodes(self.i_tr[batch_idxs])
        i_batch = jax.nn.one_hot(i_nums, self.num_elements) 
        x_batch = flatten_nodes(self.x_tr[batch_idxs])
        edges_batch = flatten_edges(self.edges_tr[batch_idxs])
        f_batch = flatten_nodes(self.f_tr[batch_idxs])
        y_batch = flatten_nodes(jnp.expand_dims(self.y_tr[batch_idxs], -1))
        return i_batch, x_batch, edges_batch, f_batch, y_batch, batch_graph_segments

if __name__ == "__main__":
    import sys
    run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
