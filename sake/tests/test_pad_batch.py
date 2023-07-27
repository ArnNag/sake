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

def test_batch_pad():
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



