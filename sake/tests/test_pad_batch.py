import pytest

def test_pad():
    import jax
    import jax.numpy as jnp
    import sake
    real_nodes = 5
    dummy_nodes = 7
    dummy_edges = 3
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
    edges_pad = jnp.pad(real_edges, ((0, dummy_edges), (0, 0)), constant_values=-1)
    init_params = model.init(jax.random.PRNGKey(2046), h_pad_one, x_pad_one, edges=edges_pad)
    h_out_one, x_out_one, v_out_one = model.apply(init_params, h_pad_one, x_pad_one, edges=edges_pad)
    h_out_two, x_out_two, v_out_two = model.apply(init_params, h_pad_two, x_pad_two, edges=edges_pad)
    assert jnp.allclose(h_out_one[:real_nodes], h_out_two[:real_nodes])
    assert jnp.allclose(x_out_one[:real_nodes], x_out_two[:real_nodes])
    assert jnp.allclose(v_out_one[:real_nodes], v_out_two[:real_nodes])
