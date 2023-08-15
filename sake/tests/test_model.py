import pytest

def test_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    import jraph
    from flax.core import frozen_dict
    model = sake.models.SAKEModel(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=None, n_node=[5], n_edge=len(edge_idxs))
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    graph = model.apply(init_params, graph)
    h = graph.nodes["h"]
    x = graph.nodes["x"]
    v = graph.nodes["v"]
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)
