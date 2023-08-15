import pytest

def test_exp_normal_smearing():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.utils.ExpNormalSmearing()
    key = jax.random.PRNGKey(2666)
    x = jax.random.normal(key=key, shape=(5, 5, 1))
    init_params = model.init(key, x)
    out = model.apply(init_params, x)
    assert out.shape == x.shape[:-1] + (model.num_rbf,)

def test_cfc_with_concatenation():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.ContinuousFilterConvolutionWithConcatenation(16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 5, 1))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 5, 4))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h = model.apply(init_params, h, x)
    assert h.shape == (5, 5, 16)

def test_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    import jraph
    from flax.core import frozen_dict
    model = sake.layers.SAKELayer(16, 16)
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

