import pytest

def test_pad():
    import jax
    import jax.numpy as jnp
    import sake
    real_nodes = 5
    dummy_nodes = 7
    dummy_edges = 3
    max_nodes = real_nodes + dummy_nodes
    model = sake.layers.SparseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(real_nodes, 3))
    x_dummy = jax.random.normal(key=jax.random.PRNGKey(2023), shape=(dummy_nodes, 3))
    x_pad = jnp.concatenate((x, x_dummy), axis=0)
    print("x:", x)
    print("x_dummy:", x_dummy)
    print("x_pad:", x_pad)
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(real_nodes, 16))
    h_dummy = jax.random.uniform(key=jax.random.PRNGKey(1776), shape=(dummy_nodes, 16))
    h_pad = jnp.concatenate((h, h_dummy), axis=0)
    print("h:", h)
    print("h_dummy:", h_dummy)
    print("h_pad:", h_pad)
    real_edges = jnp.argwhere(jnp.logical_not(jnp.identity(real_nodes)))
    edges_pad = jnp.pad(real_edges, ((0, dummy_edges), (0, 0)), constant_values=-1)
    print("real_edges:", real_edges)
    print("edges_pad:", edges_pad)
    init_params = model.init(jax.random.PRNGKey(2046), h, x, edges=real_edges)
    init_params_pad = model.init(jax.random.PRNGKey(2046), h_pad, x_pad, edges=edges_pad)
    h, x, v = model.apply(init_params, h, x, edges=real_edges)
    h_pad, x_pad, v_pad = model.apply(init_params_pad, h_pad, x_pad, edges=edges_pad)
    assert jnp.allclose(h, h_pad[:real_nodes])
    assert jnp.allclose(x, x_pad[:real_nodes])
    assert jnp.allclose(v, v_pad[:real_nodes])
