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

def test_dense_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h, x, v = model.apply(init_params, h, x)
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)

def test_sparse_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    max_nodes = 5
    dense_model = sake.layers.DenseSAKELayer(16, 16)
    sparse_model = sake.layers.SparseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(max_nodes, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(max_nodes, 16))
    edges = jnp.argwhere(jnp.logical_not(jnp.identity(max_nodes)))
    init_params_dense = dense_model.init(jax.random.PRNGKey(2046), h, x)
    init_params_sparse = sparse_model.init(jax.random.PRNGKey(2046), h, x, edges=edges, max_nodes=max_nodes)
    h_dense, x_dense, v_dense = dense_model.apply(init_params_dense, h, x)
    h_sparse, x_sparse, v_sparse = sparse_model.apply(init_params_sparse, h, x, edges=edges, max_nodes=max_nodes)

    print("x_dense:", x_dense)
    print("x_sparse:", x_sparse)
    assert h_sparse.shape == (5, 16)
    assert x_sparse.shape == (5, 3)
    assert v_sparse.shape == (5, 3)
    assert jnp.allclose(h_dense, h_sparse, atol=1e-3)
    assert jnp.allclose(x_dense, x_sparse, atol=1e-1)
    assert jnp.allclose(v_dense, v_sparse, atol=1e-1)

