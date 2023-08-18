import numpy as onp
import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
import sake
from typing import Optional
from functools import partial
from flax.core import frozen_dict

# Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
ELEMENT_MAP = onp.array([0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15])

NUM_ELEMENTS = 16


def make_graph_list(h, x, edge_idxs, f, y, num_nodes, num_edges):
    def make_graph(idx):
        graph = jraph.GraphsTuple(
                n_node=jnp.array([num_nodes[idx]]),
                n_edge=jnp.array([num_edges[idx]]),
                nodes=frozen_dict.freeze({"h": h[idx, :num_nodes[idx]], "x": x[idx, :num_nodes[idx]], "f": f[idx, :num_nodes[idx]]}),
                senders=edge_idxs[idx, :num_edges[idx], 0],
                receivers=edge_idxs[idx, :num_edges[idx], 1],
                edges=None,
                globals=y[idx]
                )
        return graph 
    return [make_graph(idx) for idx in range(len(y))]

def load_data(path, subset=-1):
    ds = onp.load(path)
    i = ELEMENT_MAP[ds["atomic_numbers"]]
    x = ds["pos"]
    edge_idxs = ds["edges"]
    f = ds["forces"]
    y = ds["formation_energy"]
    num_nodes = ds["num_nodes"]
    num_edges = ds["num_edges"]
    subset_labels = ds["subsets"]
    if subset >= 0:
        i, x, edge_idxs, f, y, num_nodes, num_edges = select(subset_labels, subset, i, x, edge_idxs, f, y, num_nodes, num_edges)

    return make_graph_list(i, x, edge_idxs, f, y, num_nodes, num_edges), y.mean(), y.std()


def select(subset_labels, subset, *fields):
    selection = (subset_labels == subset)
    return (field[selection] for field in fields)


def make_batch_loader(graph_list, seed, max_nodes, max_edges, max_graphs, num_elements):
    """
    Initialize for every epoch with a unique seed.
    """
    key = jax.random.PRNGKey(seed)
    idxs = jax.random.permutation(key, len(graph_list))
    graph_gen = jraph.dynamically_batch((graph_list[idx] for idx in idxs), n_node=max_nodes, n_edge=max_edges, n_graph=max_graphs)
    for graph in graph_gen:
        yield graph._replace(nodes=graph.nodes.copy({"h": jax.nn.one_hot(graph.nodes["h"], num_elements)}))


def partition_sum(data: jnp.ndarray,
                  partitions: jnp.ndarray,
                  sum_partitions: Optional[int] = None):
    """Compute a sum within partitions of an array.

    For example:
          data = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0])
          partitions = jnp.array([3, 2])
          partition_sum(data, partitions)
          >> DeviceArray(
          >> [6.0, 3.0],
          >> dtype=float32)

    Args:
      data: an array with the values to be summed.
      partitions: the number of nodes per graph. It is a vector of integers with
        shape ``[n_graphs]``, such that ``graph.n_node[i]`` is the number of nodes
        in the i-th graph.
      sum_partitions: the sum of n_node. If not passed, the result of this method
        is data dependent and so not ``jit``-able.

    Returns:
      The sum over partitions.
      """
    n_partitions = len(partitions)
    segment_ids = jnp.repeat(jnp.arange(n_partitions), partitions, axis=0, total_repeat_length=sum_partitions)
    return jax.ops.segment_sum(data, segment_ids, n_partitions, indices_are_sorted=True)


class SAKEEnergyModel(nn.Module):

    def setup(self):
        self.model = sake.models.SAKEModel(
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

    def __call__(self, graph):
        h = self.model(graph).nodes['h']
        y = partition_sum(h, graph.n_node, sum_partitions=len(graph.nodes['x']))
        y = self.mlp(y)
        y = self.coloring(y)
        return y


@partial(jax.jit, static_argnums=(0,))
def get_y_pred(model, params, graph):
    y_pred = model.apply(params, graph)
    return y_pred


@partial(jax.jit, static_argnums=(0,))
def get_neg_y_pred_sum(model, params, graph, x):
    graph = graph._replace(nodes=graph.nodes.copy(add_or_replace={"x": x}))
    y_pred = get_y_pred(model, params, graph)
    return -y_pred.sum()


get_f_pred = jax.jit(jax.grad(get_neg_y_pred_sum, argnums=3), static_argnums=(0,))


@partial(jax.jit, static_argnums=(0,))
def get_y_loss(model, params, graph):
    y_mask = jnp.expand_dims(jraph.get_graph_padding_mask(graph), axis=-1)
    jax.debug.print("Num real graphs: {}", jnp.sum(y_mask))
    y_pred = get_y_pred(model, params, graph)
    y_loss = jnp.abs((y_pred - graph.globals) * y_mask).sum()
    return y_loss


@partial(jax.jit, static_argnums=(0,))
def get_f_loss(model, params, graph):
    f_mask = jnp.expand_dims(jraph.get_node_padding_mask(graph), axis=-1)
    jax.debug.print("Num real nodes: {}", jnp.sum(f_mask))
    jax.debug.print("Num real edge: {}", jnp.sum(jraph.get_edge_padding_mask(graph)))
    f_pred = get_f_pred(model, params, graph, graph.nodes['x'])
    f_loss = jnp.abs((f_pred - graph.nodes['f']) * f_mask).sum()
    return f_loss
