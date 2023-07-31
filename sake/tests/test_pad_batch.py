import pytest

def test_pad():
    import jax
    import jax.numpy as jnp
    import sake
    real_nodes = 5
    dummy_nodes = 7
    dummy_edges_one = 3
    dummy_edges_two = 11
    hidden_features = 2
    max_nodes = real_nodes + dummy_nodes
    model = sake.layers.SparseSAKELayer(hidden_features, hidden_features)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(real_nodes, 3))
    x_dummy_one = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(dummy_nodes, 3))
    x_dummy_two = jax.random.normal(key=jax.random.PRNGKey(2024), shape=(dummy_nodes, 3))
    x_pad_one = jnp.concatenate((x, x_dummy_one), axis=0)
    x_pad_two = jnp.concatenate((x, x_dummy_two), axis=0)
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(real_nodes, hidden_features))
    h_dummy_one = jax.random.uniform(key=jax.random.PRNGKey(1776), shape=(dummy_nodes, hidden_features))
    h_dummy_two = jax.random.uniform(key=jax.random.PRNGKey(1777), shape=(dummy_nodes, hidden_features))
    h_pad_one = jnp.concatenate((h, h_dummy_one), axis=0)
    h_pad_two = jnp.concatenate((h, h_dummy_two), axis=0)
    real_edges = jnp.argwhere(jnp.logical_not(jnp.identity(real_nodes)))
    edges_pad_one = jnp.pad(real_edges, ((0, dummy_edges_one), (0, 0)), constant_values=-1)
    edges_pad_two = jnp.pad(real_edges, ((0, dummy_edges_two), (0, 0)), constant_values=-1)
    init_params = model.init(jax.random.PRNGKey(2046), h_pad_one, x_pad_one, edges=edges_pad_one)
    h_out_one, x_out_one, v_out_one = model.apply(init_params, h_pad_one, x_pad_one, edges=edges_pad_one)
    h_out_two, x_out_two, v_out_two = model.apply(init_params, h_pad_two, x_pad_two, edges=edges_pad_two)
    assert jnp.allclose(h_out_one[:real_nodes], h_out_two[:real_nodes])
    assert jnp.allclose(x_out_one[:real_nodes], x_out_two[:real_nodes])
    assert jnp.allclose(v_out_one[:real_nodes], v_out_two[:real_nodes])

def test_diff_pad_nodes_loss():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import SPICEBatchLoader, SparseSAKEEnergyModel, get_y_loss, get_f_loss
    real_nodes = 5
    dummy_nodes = 7
    hidden_features = 2
    max_nodes = real_nodes + dummy_nodes
    graph_segments = jnp.array([0] * real_nodes + [-1] * dummy_nodes)
    model = SparseSAKEEnergyModel(num_segments=1)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(real_nodes, 3))
    x_dummy_one = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(dummy_nodes, 3))
    x_dummy_two = jax.random.normal(key=jax.random.PRNGKey(2024), shape=(dummy_nodes, 3))
    x_pad_one = jnp.concatenate((x, x_dummy_one), axis=0)
    x_pad_two = jnp.concatenate((x, x_dummy_two), axis=0)
    h_pad_one = x_pad_one
    h_pad_two = x_pad_two
    f_pad_one = x_pad_one
    f_pad_two = x_pad_two
    real_edges = jnp.argwhere(jnp.logical_not(jnp.identity(real_nodes)))
    init_params = model.init(jax.random.PRNGKey(2046), h_pad_one, x_pad_one, edges=real_edges, graph_segments=graph_segments)
    f_loss_one = get_f_loss(model, init_params, h_pad_one, x_pad_one, edges=real_edges, f=f_pad_one, graph_segments=graph_segments) 
    f_loss_two = get_f_loss(model, init_params, h_pad_two, x_pad_two, edges=real_edges, f=f_pad_two, graph_segments=graph_segments)
    e_loss_one = get_y_loss(model, init_params, h_pad_one, x_pad_one, edges=real_edges, y=0, graph_segments=graph_segments)
    e_loss_two = get_y_loss(model, init_params, h_pad_two, x_pad_two, edges=real_edges, y=0, graph_segments=graph_segments)
    assert f_loss_one == f_loss_two
    assert e_loss_one == e_loss_two

def test_diff_pad_nodes_pred():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import SPICEBatchLoader, SparseSAKEEnergyModel, get_e_pred, get_f_pred 
    real_nodes = 5
    dummy_nodes = 7
    hidden_features = 2
    max_nodes = real_nodes + dummy_nodes
    graph_segments = jnp.array([0] * real_nodes + [-1] * dummy_nodes)
    model = SparseSAKEEnergyModel(num_segments=1)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(real_nodes, 3))
    x_dummy_one = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(dummy_nodes, 3))
    x_dummy_two = jax.random.normal(key=jax.random.PRNGKey(2024), shape=(dummy_nodes, 3))
    x_pad_one = jnp.concatenate((x, x_dummy_one), axis=0)
    x_pad_two = jnp.concatenate((x, x_dummy_two), axis=0)
    h_pad_one = x_pad_one
    h_pad_two = x_pad_two
    real_edges = jnp.argwhere(jnp.logical_not(jnp.identity(real_nodes)))
    init_params = model.init(jax.random.PRNGKey(2046), h_pad_one, x_pad_one, edges=real_edges, graph_segments=graph_segments)
    f_pred_one = get_f_pred(model, init_params, h_pad_one, x_pad_one, edges=real_edges, graph_segments=graph_segments)
    f_pred_two = get_f_pred(model, init_params, h_pad_two, x_pad_two, edges=real_edges, graph_segments=graph_segments)
    e_pred_one = get_e_pred(model, init_params, h_pad_one, x_pad_one, edges=real_edges, graph_segments=graph_segments)
    e_pred_two = get_e_pred(model, init_params, h_pad_two, x_pad_two, edges=real_edges, graph_segments=graph_segments)
    assert jnp.allclose(f_pred_one, f_pred_two)
    assert jnp.allclose(e_pred_one, e_pred_two)

def test_pad_batch():
    import sys
    import jax
    import jax.numpy as jnp
    import sake
    sys.path.append('../../scripts/spice')
    from sparse import SPICEBatchLoader

    num_elements = 17
    geom_features = 3
    y_features = 1
    num_graphs_tr = 997
    num_nodes_tr = 7
    num_edges_tr = 5
    max_graphs_batch = 29
    max_nodes_batch = 53
    max_edges_batch = 59
    key = jax.random.PRNGKey(2666)
    nodes_per_graph = jax.random.randint(key, shape=(num_graphs_tr,), minval=2, maxval=num_nodes_tr)
    edges_per_graph = jax.random.randint(key, shape=(num_graphs_tr,), minval=1, maxval=num_edges_tr)
    i_tr = []
    x_tr = []
    edges_tr = []
    f_tr = []
    y_tr = []

    for i, (num_nodes, num_edges) in enumerate(zip(nodes_per_graph, edges_per_graph)):
        key = jax.random.PRNGKey(i)
        i = jnp.pad(jax.random.randint(key, shape=(num_nodes,), minval=0, maxval=num_elements-1), ((0, num_nodes_tr - num_nodes)))
        x = jnp.pad(jax.random.uniform(key, shape=(num_nodes, geom_features)), ((0, num_nodes_tr - num_nodes), (0, 0)))
        all_edges = jnp.argwhere(jnp.logical_not(jnp.identity(num_nodes))) 
        edges = jnp.pad(all_edges[jax.random.permutation(key, num_edges_tr)[:num_edges]], ((0, num_edges_tr - num_edges), (0, 0)), mode='constant', constant_values=-1)
        f = jnp.pad(jax.random.uniform(key, shape=(num_nodes, geom_features)), ((0, num_nodes_tr - num_nodes), (0, 0)))
        y = jnp.pad(jax.random.uniform(key, shape=(num_nodes, y_features)), ((0, num_nodes_tr - num_nodes), (0, 0)))
        i_tr.append(i)
        x_tr.append(x)
        edges_tr.append(edges)
        f_tr.append(f)
        y_tr.append(y)
    i_tr = jnp.array(i_tr)
    x_tr = jnp.array(x_tr)
    edges_tr = jnp.array(edges_tr)
    f_tr = jnp.array(f_tr)
    y_tr = jnp.array(y_tr)
    assert(i_tr.shape == (num_graphs_tr, num_nodes_tr))
    assert(x_tr.shape == (num_graphs_tr, num_nodes_tr, geom_features))
    assert(edges_tr.shape == (num_graphs_tr, num_edges_tr, 2))
    assert(f_tr.shape == (num_graphs_tr, num_nodes_tr, geom_features))
    assert(y_tr.shape == (num_graphs_tr, num_nodes_tr, y_features))
            
    print("nodes_per_graph", nodes_per_graph)
    print("edges_per_graph", edges_per_graph)
    loader = SPICEBatchLoader(i_tr, x_tr, edges_tr, f_tr, y_tr, nodes_per_graph, edges_per_graph, 1776, max_edges_batch, max_nodes_batch, max_graphs_batch, num_elements)

def test_basic():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from sparse import SPICEBatchLoader
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    x_tr = i_tr
    f_tr = i_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1776, max_edges=4, max_nodes=7, max_graphs=3, num_elements=13)
    i, x, edges, f, y, graph_segments = loader.get_batch(0)
    assert jnp.allclose(i, jnp.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    assert jnp.allclose(x, jnp.array([4., 8., 7., 11., 13, 0., 0.]))
    assert jnp.allclose(edges, jnp.array([[0., 1.], [2., 3.], [3., 4.], [-1., -1.]]))
    assert jnp.allclose(f, jnp.array([4., 8., 7., 11., 13, 0., 0.]))
    assert jnp.allclose(y, jnp.array([[17.], [20.], [0.]]))
    assert jnp.allclose(graph_segments, jnp.array([0., 0., 1., 1., 1., -1., -1.]))

def test_max_graphs_reached():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from sparse import SPICEBatchLoader
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    x_tr = i_tr
    f_tr = i_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1776, max_edges=4, max_nodes=7, max_graphs=1, num_elements=13)
    i0, x0, edges0, f0, y0, graph_segments0 = loader.get_batch(0)
    i1, x1, edges1, f1, y1, graph_segments1 = loader.get_batch(1)
    assert jnp.allclose(i0, jnp.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    assert jnp.allclose(i1, jnp.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    assert jnp.allclose(x0, jnp.array([4., 8., 0., 0., 0, 0., 0.]))
    assert jnp.allclose(x1, jnp.array([7., 11., 13., 0., 0., 0., 0.]))
    assert jnp.allclose(edges0, jnp.array([[0., 1.], [-1., -1.], [-1., -1.], [-1., -1.]]))
    assert jnp.allclose(edges1, jnp.array([[0., 1.], [1., 2.], [-1., -1.], [-1., -1.]]))
    assert jnp.allclose(f0, jnp.array([4., 8., 0., 0., 0, 0., 0.]))
    assert jnp.allclose(f1, jnp.array([7., 11., 13., 0., 0., 0., 0.]))
    assert jnp.allclose(y0, jnp.array([[17.]]))
    assert jnp.allclose(y1, jnp.array([[20.]]))
    assert jnp.allclose(graph_segments0, jnp.array([0., 0., -1., -1., -1., -1., -1.]))
    assert jnp.allclose(graph_segments1, jnp.array([0., 0., 0., -1., -1., -1., -1.]))

def test_max_nodes_reached():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from sparse import SPICEBatchLoader
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    x_tr = i_tr
    f_tr = i_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    loader = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1776, max_edges=5, max_nodes=4, max_graphs=3, num_elements=13)
    i0, x0, edges0, f0, y0, graph_segments0 = loader.get_batch(0)
    i1, x1, edges1, f1, y1, graph_segments1 = loader.get_batch(1)
    assert jnp.allclose(i0, jnp.array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    assert jnp.allclose(i1, jnp.array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    assert jnp.allclose(x0, jnp.array([4., 8., 0., 0.]))
    assert jnp.allclose(x1, jnp.array([7., 11., 13., 0.]))
    assert jnp.allclose(edges0, jnp.array([[0., 1.], [-1., -1.], [-1., -1.], [-1., -1.], [-1., -1.]]))
    assert jnp.allclose(edges1, jnp.array([[0., 1.], [1., 2.], [-1., -1.], [-1., -1.], [-1., -1.]]))
    assert jnp.allclose(f0, jnp.array([4., 8., 0., 0.]))
    assert jnp.allclose(f1, jnp.array([7., 11., 13., 0.]))
    assert jnp.allclose(y0, jnp.array([[17.], [0.], [0.]]))
    assert jnp.allclose(y1, jnp.array([[20.], [0.], [0.]]))
    assert jnp.allclose(graph_segments0, jnp.array([0., 0., -1., -1.]))
    assert jnp.allclose(graph_segments1, jnp.array([0., 0., 0., -1.]))


def test_graph_order():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import SPICEBatchLoader, SparseSAKEEnergyModel, get_e_pred, get_f_pred
    jax.config.update("jax_debug_nans", True)
    key = jax.random.PRNGKey(0)
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    num_graphs = i_tr.shape[0]
    num_nodes = i_tr.shape[1]
    x_tr = jax.random.uniform(key, shape=(num_graphs, num_nodes, 3))
    f_tr = x_tr
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_nodes_tr = jnp.array([3, 2])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    max_edges = 4
    max_nodes = 7
    max_graphs = 3
    num_elements = 13
    loader0 = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=0, max_edges=max_edges, max_nodes=max_nodes, max_graphs=max_graphs, num_elements=num_elements)
    loader1 = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1, max_edges=max_edges, max_nodes=max_nodes, max_graphs=max_graphs, num_elements=num_elements)
    assert(loader0.batch_list[0] == [1, 0])
    assert(loader1.batch_list[0] == [0, 1])
    i0, x0, edges0, f0, y0, graph_segments0 = loader0.get_batch(0)
    i1, x1, edges1, f1, y1, graph_segments1 = loader1.get_batch(0)
    model = SparseSAKEEnergyModel(num_segments=max_graphs)
    variables = model.init(key, i0, x0, edges0, graph_segments0)
    e_pred0 = get_e_pred(model, variables, i0, x0, edges0, graph_segments0) 
    e_pred1 = get_e_pred(model, variables, i1, x1, edges1, graph_segments1)
    f_pred0 = get_f_pred(model, variables, i0, x0, edges0, graph_segments0)
    f_pred1 = get_f_pred(model, variables, i1, x1, edges1, graph_segments1)
    print("f_preds:", f_pred0, f_pred1)
    assert(jnp.allclose(e_pred0[0], e_pred1[1]))
    assert(jnp.allclose(e_pred0[1], e_pred1[0]))
    assert(jnp.allclose(f_pred0[0:2], f_pred1[3:5]))
    assert(jnp.allclose(f_pred0[2:5], f_pred1[0:3]))
    
def test_graph_order_loss():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import SPICEBatchLoader, SparseSAKEEnergyModel, get_y_loss, get_f_loss
    key = jax.random.PRNGKey(0)
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    num_graphs = i_tr.shape[0]
    num_nodes = i_tr.shape[1]
    x_tr = jax.random.uniform(key, shape=(num_graphs, num_nodes, 3))
    f_tr = x_tr
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_nodes_tr = jnp.array([3, 2])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    max_edges = 4
    max_nodes = 7
    max_graphs = 3
    num_elements = 13
    loader0 = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=0, max_edges=max_edges, max_nodes=max_nodes, max_graphs=max_graphs, num_elements=num_elements)
    loader1 = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1, max_edges=max_edges, max_nodes=max_nodes, max_graphs=max_graphs, num_elements=num_elements)
    i0, x0, edges0, f0, y0, graph_segments0 = loader0.get_batch(0)
    i1, x1, edges1, f1, y1, graph_segments1 = loader1.get_batch(0)
    model = SparseSAKEEnergyModel(num_segments=max_graphs)
    variables = model.init(key, i0, x0, edges0, graph_segments0)
    e_loss0 = get_y_loss(model, variables, i0, x0, edges0, y0, graph_segments0) 
    e_loss1 = get_y_loss(model, variables, i1, x1, edges1, y1, graph_segments1)
    f_loss0 = get_f_loss(model, variables, i0, x0, edges0, f0, graph_segments0)
    f_loss1 = get_f_loss(model, variables, i1, x1, edges1, f1, graph_segments1)
    assert(e_loss0 == e_loss1)
    assert(f_loss0 == f_loss1)

def test_max_graphs_reached_loss():
    import jax
    import jax.numpy as jnp
    import sake
    import sys
    sys.path.append('../../scripts/spice')
    from utils import SPICEBatchLoader, SparseSAKEEnergyModel, get_y_loss, get_f_loss
    i_tr = jnp.array([[7, 11, 13, 0, 0], [4, 8, 0, 0, 0]])
    key = jax.random.PRNGKey(0)
    num_graphs = i_tr.shape[0]
    num_nodes = i_tr.shape[1]
    x_tr = jax.random.uniform(key, shape=(num_graphs, num_nodes, 3))
    f_tr = x_tr
    num_nodes_tr = jnp.array([3, 2])
    edges_tr = jnp.array([[[0, 1], [1, 2], [-1, -1], [-1, -1], [-1, -1]], [[0, 1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
    num_edges_tr = jnp.array([2, 1])
    y_tr = jnp.array([20, 17])
    max_graphs_split = 1
    max_graphs_unsplit = 5
    model_split = SparseSAKEEnergyModel(num_segments=max_graphs_split)
    model_unsplit = SparseSAKEEnergyModel(num_segments=max_graphs_unsplit)
    loader_split = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1776, max_edges=4, max_nodes=7, max_graphs=max_graphs_split, num_elements=13)
    loader_unsplit = SPICEBatchLoader(i_tr=i_tr, x_tr=x_tr, edges_tr=edges_tr, f_tr=f_tr, y_tr=y_tr, num_nodes_tr=num_nodes_tr, num_edges_tr=num_edges_tr, seed=1776, max_edges=4, max_nodes=7, max_graphs=max_graphs_unsplit, num_elements=13)
    i0_split, x0_split, edges0_split, f0_split, y0_split, graph_segments0_split = loader_split.get_batch(0)
    variables = model_split.init(key, i0_split, x0_split, edges=edges0_split, graph_segments=graph_segments0_split)
    total_f_loss_split = 0
    total_e_loss_split = 0
    for idx in range(len(loader_split)):
        i, x, edges, f, y, graph_segments = loader_split.get_batch(idx)
        total_f_loss_split += get_f_loss(model_split, variables, i, x, edges, f, graph_segments)
        total_e_loss_split += get_y_loss(model_split, variables, i, x, edges, y, graph_segments)

    total_f_loss_unsplit = 0
    total_e_loss_unsplit = 0
    for idx in range(len(loader_unsplit)):
        i, x, edges, f, y, graph_segments = loader_unsplit.get_batch(i)
        total_f_loss_unsplit += get_f_loss(model_unsplit, variables, i, x, edges, f, graph_segments)
        total_e_loss_unsplit += get_y_loss(model_unsplit, variables, i, x, edges, y, graph_segments)

    assert(total_f_loss_split == total_f_loss_unsplit)
    assert(total_e_loss_split == total_e_loss_unsplit)



def test_segment_softmax():
    import jax.numpy as jnp
    from utils import segment_softmax
    att = jnp.array([[ 0.05218441, -0.11108486,  0.06595716,  0.0130074 ],
     [ 0.13529724,  0.01634026, -0.03200634, -0.00883934],
     [ 0.02268515,  0.07742056, -0.0038277,  -0.1017934 ],
     [ 0.0175745,  -0.08937339,  0.04781523,  0.07290111]])
    segments = jnp.array([1, 3, 4, -1])
    att = segment_softmax(att, segments)

