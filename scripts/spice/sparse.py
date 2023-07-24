import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
from utils import ELEMENT_MAP, NUM_ELEMENTS, select

def run(prefix, max_nodes=997, max_edges=14983, max_graphs=53, e_loss_factor=0, subset=None):
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

        def __call__(self, i, x, edges, graph_segments):
            h = self.model(i, x, edges=edges)[0]
            y = jax.ops.segment_sum(h, graph_segments, num_segments=max_graphs)
            y = self.mlp(y)
            return y

    model = Model()


    @jax.jit
    def get_e_pred(params, i, x, edges, graph_segments):
        e_pred = model.apply(params, i, x, edges, graph_segments)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x, edges, graph_segments):
        e_pred = get_e_pred(params, i, x, edges, graph_segments)
        return -e_pred.sum()

    get_f_pred = jax.jit(jax.grad(get_e_pred_sum, argnums=2))

    def loss_fn(params, i, x, edges, f, y, graph_segments):
        real_nodes = jnp.array(jnp.not_equal(graph_segments, -1), dtype=int)
        jax.debug.print("Num real nodes: {}", jnp.sum(real_nodes))
        e_mask = jax.ops.segment_sum(real_nodes, graph_segments, num_segments=max_graphs) != 0
        jax.debug.print("Num real graphs: {}", jnp.sum(e_mask))
        jax.debug.print("Num real edge: {}", jnp.sum(edges[:,1] != -1))
        f_mask = jnp.expand_dims(real_nodes, -1)
        e_pred = get_e_pred(params, i, x, edges, graph_segments) * e_mask
        f_pred = get_f_pred(params, i, x, edges, graph_segments) * f_mask
        e_loss = jnp.abs(e_pred - y).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * e_loss_factor

    @jax.jit
    def step_with_loss(state, i, x, edges, f, y, graph_segments):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, edges, f, y, graph_segments)
        state = state.apply_gradients(grads=grads)
        return state
    
    def epoch(state, i_tr, x_tr, edges_tr, f_tr, y_tr):
        loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=state.step, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs, num_elements=NUM_ELEMENTS)

        for idx in tqdm.tqdm(range(len(loader))):
            i, x, edges, f, y, graph_segments = loader.get_batch(idx)  
            state = step_with_loss(state, i, x, edges, f, y, graph_segments)

        return state

    init_loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=2666, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs, num_elements=NUM_ELEMENTS)
    i0, x0, edges0, _, __, graph_segments0 = init_loader.get_batch(0)

    key = jax.random.PRNGKey(2666)
    params = model.init(key, i0, x0, edges0, graph_segments0)

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
        print(state.opt_state.notfinite_count)
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint(f"_sparse_{prefix}eloss_{e_loss_factor:.0e}_subset_{subset}", target=state, keep_every_n_steps=10, step=idx_batch)

'''
Initialize for every epoch with a unique seed.
'''
class SPICEBatchLoader:

    def __init__(self, i_tr, x_tr, edges_tr, f_tr, y_tr, num_nodes_tr, num_edges_tr, seed, max_edges, max_nodes, max_graphs, num_elements):
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
        self.batch_list, self.graph_segments = self._make_batch_list()

    def __len__(self):
        return len(self.batch_list)

    def _make_batch_list(self):
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
                batch_graph_segments.extend([len(batch_idxs)] * self.num_nodes_tr[tr_idx].item())
                batch_idxs.append(tr_idx)
                total_graphs_added += 1
            batch_list.append(batch_idxs)
            batch_graph_segments.extend([-1] * (self.max_nodes - len(batch_graph_segments)))
            graph_segment_list.append(batch_graph_segments)
        # print("num_nodes_tr:", self.num_nodes_tr)
        # print("num_edges_tr:", self.num_edges_tr)
        # print("batch_list:", batch_list)
        # print("graph_segment_list:", graph_segment_list)
        return batch_list, jnp.array(graph_segment_list)


    def get_batch(self, batch_num):
        batch_idxs = self.batch_list[batch_num]
        batch_graph_segments = self.graph_segments[batch_num]
        batch_num_nodes = self.num_nodes_tr[batch_idxs]
        batch_num_edges = self.num_edges_tr[batch_idxs]

        def flatten_data(batch_data, batch_num_data, max_num_data, fill_value, offsets):
            flattened_data = jnp.full((max_num_data, *batch_data.shape[2:]), fill_value, dtype=batch_data.dtype)
            data_flattened = 0
            for i, (graph_data, graph_num_data) in enumerate(zip(batch_data, batch_num_data)):
                next_data_idx = data_flattened + graph_num_data
                data_fill = graph_data[:graph_num_data]
                if offsets is not None:
                    data_fill += offsets[i]
                flattened_data = flattened_data.at[data_flattened:next_data_idx].set(graph_data[:graph_num_data])

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
        y_batch = jnp.expand_dims(jnp.pad(self.y_tr[batch_idxs], (0, self.max_graphs - len(batch_idxs))), -1)
        return i_batch, x_batch, edges_batch, f_batch, y_batch, batch_graph_segments

if __name__ == "__main__":
    import sys
    run(sys.argv[1], e_loss_factor=float(sys.argv[2]), subset=int(sys.argv[3]))
