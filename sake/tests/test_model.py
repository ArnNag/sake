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
    y = jax.random.normal(key=jax.random.PRNGKey(2048), shape=(1, 1))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None, "f": f}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=y, n_node=jnp.array([5]), n_edge=len(edge_idxs))
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    y_pred = get_y_pred(model, init_params, graph)
    assert(y_pred.shape == (1,1))

def test_y_loss():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    import jraph
    from flax.core import frozen_dict
    sys.path.append('../../scripts/spice')
    from utils import SAKEEnergyModel, get_y_loss, get_f_loss
    model = SAKEEnergyModel()
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    f = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(5, 3))
    y = jax.random.normal(key=jax.random.PRNGKey(2048), shape=(1, 1))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None, "f": f}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=y, n_node=jnp.array([5]), n_edge=jnp.array([len(edge_idxs)]))
    graph = jraph.pad_with_graphs(graph, 10, 50)
    init_params = model.init(jax.random.PRNGKey(2046), graph)
    y_loss = get_y_loss(model, init_params, graph)
    print("y_loss:", y_loss)
    assert(y_loss.shape == ())
    f_loss = get_f_loss(model, init_params, graph)
    print("f_loss:", f_loss)
    assert(f_loss.shape == ())

def test_pad():
    """
    Check that different numbers of node and edge padding don't change the value of the loss"
    """
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    import jraph
    from flax.core import frozen_dict
    sys.path.append('../../scripts/spice')
    from utils import SAKEEnergyModel, get_y_loss, get_f_loss
    model = SAKEEnergyModel()
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    f = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(5, 3))
    y = jax.random.normal(key=jax.random.PRNGKey(2048), shape=(1, 1))
    edge_idxs = jnp.argwhere(jnp.logical_not(jnp.identity(5)))
    graph = jraph.GraphsTuple(nodes=frozen_dict.freeze({"x": x, "h": h, "v": None, "f": f}), edges=None, senders=edge_idxs[:,0], receivers=edge_idxs[:,1], globals=y, n_node=jnp.array([5]), n_edge=jnp.array([len(edge_idxs)]))
    graph_one = jraph.pad_with_graphs(graph, 10, 50)
    graph_two = jraph.pad_with_graphs(graph, 15, 50)
    graph_three = jraph.pad_with_graphs(graph, 10, 60)
    init_params = model.init(jax.random.PRNGKey(2046), graph_one)
    y_loss_one = get_y_loss(model, init_params, graph_one)
    y_loss_two = get_y_loss(model, init_params, graph_two)
    y_loss_three = get_y_loss(model, init_params, graph_three)
    assert(y_loss_one == y_loss_two == y_loss_three)
    f_loss_one = get_f_loss(model, init_params, graph_one)
    f_loss_two = get_f_loss(model, init_params, graph_two)
    f_loss_three = get_f_loss(model, init_params, graph_three)
    assert(f_loss_one == f_loss_two == f_loss_three)

def test_max_nodes_reached():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import make_graph_list, make_batch_loader
    i_tr = jnp.array([[7, 11, 12, 0, 0], [4, 8, 0, 0, 0]])
    x_tr = i_tr
    f_tr = i_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([[20], [17]])
    graph_list = make_graph_list(i_tr, x_tr, edges_tr, f_tr, y_tr, num_nodes_tr, num_edges_tr)
    batch_loader = make_batch_loader(graph_list, seed=1776, max_edges=5, max_nodes=4, max_graphs=3, num_elements=13)
    graph_one = next(batch_loader)
    graph_two = next(batch_loader)
    assert jnp.allclose(graph_one.nodes['h'][:2], jnp.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
       ]))
    assert jnp.allclose(graph_two.nodes['h'][:3], jnp.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
       ]))
    assert jnp.allclose(graph_one.nodes['x'], jnp.array([4., 8., 0., 0.]))
    assert jnp.allclose(graph_two.nodes['x'], jnp.array([7., 11., 12., 0.]))
    assert jnp.allclose(graph_one.senders[:1], jnp.array([0]))
    assert jnp.allclose(graph_one.receivers[:1], jnp.array([1]))
    assert jnp.allclose(graph_two.senders[:2], jnp.array([0, 1]))
    assert jnp.allclose(graph_two.receivers[:2], jnp.array([1, 2]))
    assert jnp.allclose(graph_one.nodes['f'][:2], jnp.array([4., 8.]))
    assert jnp.allclose(graph_two.nodes['f'][:3], jnp.array([7., 11., 12.]))
    assert jnp.allclose(graph_one.globals[0], jnp.array([[17.]]))
    assert jnp.allclose(graph_two.globals[0], jnp.array([[20.]]))
    assert jnp.allclose(graph_one.n_node[0], jnp.array([2]))
    assert jnp.allclose(graph_two.n_node[0], jnp.array([3]))
    assert jnp.allclose(graph_one.n_edge[0], jnp.array([1]))
    assert jnp.allclose(graph_two.n_edge[0], jnp.array([2]))

def test_max_nodes_not_reached():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import make_graph_list, make_batch_loader
    i_tr = jnp.array([[7, 11, 12, 0, 0], [4, 8, 0, 0, 0]])
    x_tr = i_tr
    f_tr = i_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([[20], [17]])
    graph_list = make_graph_list(i_tr, x_tr, edges_tr, f_tr, y_tr, num_nodes_tr, num_edges_tr)
    batch_loader = make_batch_loader(graph_list, seed=1776, max_edges=5, max_nodes=6, max_graphs=3, num_elements=13)
    graph_one = next(batch_loader)
    assert jnp.allclose(graph_one.nodes['h'][:5], jnp.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
       ]))
    assert jnp.allclose(graph_one.nodes['x'][:5], jnp.array([4., 8., 7., 11., 12.]))
    assert jnp.allclose(graph_one.senders[:3], jnp.array([0, 2, 3]))
    assert jnp.allclose(graph_one.receivers[:3], jnp.array([1, 3, 4]))
    assert jnp.allclose(graph_one.nodes['f'][:5], jnp.array([4., 8., 7., 11., 12.]))
    assert jnp.allclose(graph_one.globals[:2], jnp.array([[17., 20.]]))
    assert jnp.allclose(graph_one.n_node[:2], jnp.array([2, 3]))
    assert jnp.allclose(graph_one.n_edge[:2], jnp.array([1, 2]))
