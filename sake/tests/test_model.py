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

def test_same_batched():
    '''
    Test that the model produces the same output when the graph is batched vs when it is not batched.
    '''
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import make_graph_list, make_batch_loader, get_y_pred, get_y_loss, SAKEEnergyModel
    jax.disable_jit()
    i_tr = jnp.array([[7, 11, 12, 0, 0], [4, 8, 0, 0, 0]])
    num_graphs_load = i_tr.shape[0]
    num_nodes_load = i_tr.shape[1]
    x_tr_shape = jnp.array([num_graphs_load, num_nodes_load, 3])
    x_tr = jnp.arange(jnp.prod(x_tr_shape)).reshape(x_tr_shape)
    f_tr = x_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([[20], [17]])
    graph_list = make_graph_list(i_tr, x_tr, edges_tr, f_tr, y_tr, num_nodes_tr, num_edges_tr)
    assert(graph_list[0].nodes['x'].shape == (num_nodes_tr[0], 3)) 
    unbatched_max_nodes = 4
    unbatched_max_edges = 17
    unbatched_max_graphs = 7
    unbatched_batch_loader = make_batch_loader(graph_list, seed=1776, max_edges=unbatched_max_edges, max_nodes=unbatched_max_nodes, max_graphs=unbatched_max_graphs, num_elements=13)
    unbatched_graph_one = next(unbatched_batch_loader)
    assert(unbatched_graph_one.nodes['x'].shape == (unbatched_max_nodes,3))
    assert(unbatched_graph_one.nodes['h'].shape == (unbatched_max_nodes,13))
    assert(unbatched_graph_one.nodes['f'].shape == (unbatched_max_nodes,3))
    assert(unbatched_graph_one.globals.shape == (unbatched_max_graphs,))
    unbatched_graph_two = next(unbatched_batch_loader)
    batched_max_nodes = 27
    batched_max_edges = 5
    batched_max_graphs = 3
    batched_batch_loader = make_batch_loader(graph_list, seed=1776, max_edges=batched_max_edges, max_nodes=batched_max_nodes, max_graphs=batched_max_graphs, num_elements=13)
    batched_graph = next(batched_batch_loader)
    assert(batched_graph.nodes['x'].shape == (batched_max_nodes, 3))
    assert(batched_graph.nodes['h'].shape == (batched_max_nodes, 13))
    assert(batched_graph.nodes['f'].shape == (batched_max_nodes, 3))
    assert(batched_graph.globals.shape == (batched_max_graphs,))
    model = SAKEEnergyModel()
    seed = 2046
    init_params = model.init(jax.random.PRNGKey(seed), unbatched_graph_one)
    index = jax.random.permutation(jax.random.PRNGKey(seed), 2)
    print("after init")
    batched_y_pred = get_y_pred(model, init_params, batched_graph)
    unbatched_y_pred_one = get_y_pred(model, init_params, unbatched_graph_one)
    unbatched_y_pred_two = get_y_pred(model, init_params, unbatched_graph_two)
    assert jnp.allclose(unbatched_y_pred_one[0], batched_y_pred[index[0]])
    assert jnp.allclose(unbatched_y_pred_two[0], batched_y_pred[index[1]])

    print("batched_y_pred:", batched_y_pred)
    print("batched_graph.globals:", batched_graph.globals)
    print("unbatched_y_pred_one:", unbatched_y_pred_one)
    print("unbatched_y_pred_two:", unbatched_y_pred_two)
    print("unbatched_graph_one.globals:", unbatched_graph_one.globals)
    print("unbatched_graph_two.globals:", unbatched_graph_two.globals)


    print("computing batched_y_loss")
    batched_y_loss = get_y_loss(model, init_params, batched_graph)
    print("computing unbatched_y_loss_one")
    unbatched_y_loss_one = get_y_loss(model, init_params, unbatched_graph_one)
    print("computing unbatched_y_loss_two")
    unbatched_y_loss_two = get_y_loss(model, init_params, unbatched_graph_two)
    total_unbatched_loss = unbatched_y_loss_one + unbatched_y_loss_two
    assert jnp.allclose(total_unbatched_loss, batched_y_loss)

def test_batched_graph_feat_shapes():
    import jax.numpy as jnp
    import jax
    import jraph
    import jax.tree_util as tree
    import sys
    sys.path.append('../../scripts/spice')
    from utils import make_batch_loader
    from flax.core.frozen_dict import FrozenDict
    
    # Construct a graph
    n_node_real = 7
    node_feat_dim = 3
    edge_feat_dim = 5
    global_feat_dim = 1
    node_feats = FrozenDict({"h": jnp.ones((n_node_real, node_feat_dim))})
    global_feats = jnp.ones((1, global_feat_dim))
    edge_idxs = jnp.argwhere(jnp.ones((n_node_real, n_node_real)))
    senders = edge_idxs[:, 0]
    receivers = edge_idxs[:, 1]
    n_edge_real = senders.shape[0]
    edge_feats = jnp.ones((n_edge_real, edge_feat_dim))
    graph = jraph.GraphsTuple(nodes=node_feats, edges=edge_feats, receivers=receivers, senders=senders, globals=global_feats, n_node=[n_node_real], n_edge=jnp.array([n_edge_real]))
    assert graph.nodes['h'].shape == (n_node_real, node_feat_dim)
    assert graph.edges.shape == (len(senders), edge_feat_dim)
    assert graph.globals.shape == (1, global_feat_dim)

    # Make a batch loader with the graph we made
    graphs = [graph]

    n_node_batch = n_node_real + 1
    n_edge_batch = 97
    n_graph_batch = 23
    direct_batch_loader = jraph.dynamically_batch(graphs_tuple_iterator=graphs, n_edge=n_edge_batch, n_node=n_node_batch, n_graph=n_graph_batch)
    batched_graph = next(direct_batch_loader)
    assert batched_graph.nodes['h'].shape == (n_node_batch, node_feat_dim)
    assert batched_graph.edges.shape == (n_edge_batch, edge_feat_dim) 
    assert batched_graph.globals.shape == (n_graph_batch, global_feat_dim)

    num_elements=13
    my_batch_loader = make_batch_loader(graphs, seed=1776, max_edges=n_edge_batch, max_nodes=n_node_batch, max_graphs=n_graph_batch, num_elements=num_elements)
    my_batched_graph = next(my_batch_loader)
    assert my_batched_graph.nodes['h'].shape == (n_node_batch, node_feat_dim, num_elements)
    assert my_batched_graph.edges.shape == (n_edge_batch, edge_feat_dim) 
    assert my_batched_graph.globals.shape == (n_graph_batch, global_feat_dim)

