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
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=None, n_node=jnp.array([5]), n_edge=len(edge_idxs))
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    graph = model.apply(init_params, graph)
    h = graph.nodes["h"]
    x = graph.nodes["x"]
    v = graph.nodes["v"]
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)

def test_energy_model():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    import jraph
    from flax.core import frozen_dict
    sys.path.append('../../scripts/spice')
    from utils import SAKEEnergyModel
    model = SAKEEnergyModel()
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=None, n_node=jnp.array([5]), n_edge=len(edge_idxs))
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    energy = model.apply(init_params, graph)
    assert(energy.shape == (1,1))

def test_jit_model():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    import jraph
    from flax.core import frozen_dict
    sys.path.append('../../scripts/spice')
    from utils import SAKEEnergyModel, get_y_pred
    model = SAKEEnergyModel()
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    f = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(5, 3))
    y = jax.random.normal(key=jax.random.PRNGKey(2048), shape=(5, 1))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None, "f": f, "y": y}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=None, n_node=jnp.array([5]), n_edge=len(edge_idxs))
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    y_pred = get_y_pred(model, init_params, graph)
    assert(y_pred.shape == (1,1))

