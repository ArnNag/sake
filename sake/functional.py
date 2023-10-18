import jax
import jax.numpy as jnp
from typing import Union, Iterable, Mapping, Any

EPSILON = 1e-5
INF = 1e5


def get_x_minus_xt(graph):
    x = graph.nodes['x']
    return x[graph.receivers] - x[graph.senders]  # shape: (n_edges, 3)


def get_x_minus_xt_norm(
    x_minus_xt,
    epsilon: float = EPSILON,
):
    x_minus_xt_norm = (
        jax.nn.relu((x_minus_xt ** 2).sum(axis=-1, keepdims=True))
        + epsilon
    ) ** 0.5

    return x_minus_xt_norm


def get_h_cat_ht(graph):
    h = graph.nodes['h']
    return jnp.concatenate([h[graph.senders], h[graph.receivers]], axis=-1)
