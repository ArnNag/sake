import numpy as onp
import jax
import jax.numpy as jnp
import flax.linen as nn
import sake
from functools import partial

# Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
ELEMENT_MAP = onp.array([ 0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15]) 

NUM_ELEMENTS = 16

def load_data(path, subset=-1):
    ds = onp.load(path)
    i = ELEMENT_MAP[ds["atomic_numbers"]]
    x = ds["pos"]
    edges = ds["edges"]
    f = ds["forces"]
    y = ds["formation_energy"]
    num_nodes = ds["num_nodes"]
    num_edges = ds["num_edges"]
    subset_labels = ds["subsets"]
    if subset >= 0:
        i, x, edges, f, y, num_nodes, num_edges = select(subset_labels, subset, i, x, edges, f, y, num_nodes, num_edges)
    return i, x, edges, f, y, num_nodes, num_edges

def select(subset_labels, subset, *fields):
    selection = (subset_labels == subset)
    return (field[selection] for field in fields)

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
        batch_idxs = jnp.array(self.batch_list[batch_num])
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
                    data_fill = data_fill + offsets[i]
                flattened_data = flattened_data.at[data_flattened:next_data_idx].set(data_fill)

                data_flattened = next_data_idx
            return flattened_data

        def flatten_nodes(batch_data):
            return flatten_data(batch_data, batch_num_nodes, self.max_nodes, 0, None)

        def flatten_edges(batch_data):
            return flatten_data(batch_data, batch_num_edges, self.max_edges, -1, jnp.cumsum(jnp.concatenate([jnp.array([0]), batch_num_nodes[:-1]])))

        i_nums = flatten_nodes(self.i_tr[batch_idxs])
        i_batch = jax.nn.one_hot(i_nums, self.num_elements) 
        x_batch = flatten_nodes(self.x_tr[batch_idxs])
        edges_batch = flatten_edges(self.edges_tr[batch_idxs])
        f_batch = flatten_nodes(self.f_tr[batch_idxs])
        y_batch = jnp.expand_dims(jnp.pad(self.y_tr[batch_idxs], (0, self.max_graphs - len(batch_idxs))), -1)
        return i_batch, x_batch, edges_batch, f_batch, y_batch, batch_graph_segments

@partial(jax.jit, static_argnums=(0,))
def get_e_pred(model, params, i, x, edges, graph_segments):
    e_pred = model.apply(params, i, x, edges, graph_segments)
    return e_pred

class SparseSAKEEnergyModel(nn.Module):
    num_segments: int

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

        self.mean = self.variable("coloring", "mean", lambda: 0.)
        self.std = self.variable("coloring", "std", lambda: 1.)
        self.coloring = lambda y: self.std.value * y + self.mean.value

    def __call__(self, i, x, edges, graph_segments):
        h = self.model(i, x, edges=edges)[0]
        y = jax.ops.segment_sum(h, graph_segments, self.num_segments)
        y = self.mlp(y)
        y = self.coloring(y)
        return y

@partial(jax.jit, static_argnums=(0,))
def get_neg_e_pred_sum(model, params, i, x, edges, graph_segments):
    e_pred = get_e_pred(model, params, i, x, edges, graph_segments)
    return -e_pred.sum()

get_f_pred = jax.jit(jax.grad(get_neg_e_pred_sum, argnums=3), static_argnums=(0,))


@partial(jax.jit, static_argnums=(0,))
def get_y_loss(model, params, i, x, edges, y, graph_segments):
    e_mask = jax.ops.segment_sum(jnp.ones_like(graph_segments), graph_segments, num_segments=model.num_segments) > 0
    jax.debug.print("Num real graphs: {}", jnp.sum(e_mask))
    e_pred = get_e_pred(model, params, i, x, edges, graph_segments) * e_mask
    e_loss = jnp.abs(e_pred - y).mean()
    return e_loss

@partial(jax.jit, static_argnums=(0,))
def get_f_loss(model, params, i, x, edges, f, graph_segments):
    f_mask = jnp.expand_dims(jnp.array(jnp.not_equal(graph_segments, -1), dtype=int), -1)
    jax.debug.print("Num real nodes: {}", jnp.sum(f_mask))
    jax.debug.print("Num real edge: {}", jnp.sum(edges[:,1] != -1))
    f_pred = get_f_pred(model, params, i, x, edges, graph_segments) * f_mask
    f_loss = jnp.abs(f_pred - f).mean()
    return f_loss



