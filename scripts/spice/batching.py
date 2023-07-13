import jax
import jax.numpy as jnp


def message_passing(edges, data, num_segments):
    src = edges[:,0]
    dst = edges[:,1]
    return jax.ops.segment_sum(data[src], dst, num_segments=num_segments)


# Two graphs
def two_graphs_test():
    batch_nodes = 100
    x_nodes = 5
    x_edges = 4
    y_nodes = 6
    y_edges = 7
    hidden = 9
    data_max = 11

    key = jax.random.PRNGKey(0)
    data_x = jax.random.randint(key, shape=(x_nodes, hidden), minval=0, maxval=data_max)
    data_y = jax.random.randint(key, shape=(y_nodes, hidden), minval=0, maxval=data_max)
    idxs_x = jax.random.randint(key, shape=(x_edges, 2), minval=0, maxval=x_nodes)
    idxs_y = jax.random.randint(key, shape=(y_edges, 2), minval=0, maxval=y_nodes)
    batched_data = jnp.concatenate([data_x, data_y], axis=0)
    offset = x_nodes
    batched_idxs = jnp.concatenate([idxs_x, idxs_y + offset], axis=0)



    batched_result = message_passing(batched_idxs, batched_data)
    unbatched_result = jnp.concatenate([message_passing(idxs_x, data_x), message_passing(idxs_y, data_y)])
    passed = jnp.array_equal(batched_result, unbatched_result)
    print(passed)

# Implement batch function that takes list of idxs, data
# write unit test

def batch_message_passing(batch_idxs, batch_data):
    offset = 0
    results = []
    batch_node_pos = jnp.logical_not(jnp.all(batch_data == 0, axis=-1)) # edge case: what if all hidden values happen to be 0 for a real node
    batch_edge_pos = jnp.logical_not(jnp.all(batch_idxs == -1, axis=-1)) 
    batch_num_nodes = jnp.sum(batch_node_pos, axis=-1)
    # print("batch_num_nodes:", batch_num_nodes)
    batch_cumsum = jnp.cumsum(batch_num_nodes, axis=-1)
    batch_offset = jnp.zeros_like(batch_cumsum)
    batch_offset = batch_offset.at[1:].set(batch_cumsum[:-1])
    # print("batch_offset:", batch_offset)
    # print(jnp.expand_dims(batch_offset, -1).shape)
    # print((batch_idxs[:,:,1] != -1).shape)
    flattened_idxs = (batch_idxs + jnp.expand_dims(jnp.expand_dims(batch_offset, -1) * batch_edge_pos, -1)).reshape(batch_idxs.shape[0] * batch_idxs.shape[1], 2)[batch_edge_pos.flatten()]
    print("batch_idxs:", batch_idxs)
    print("flattened_idxs:", flattened_idxs)
    flattened_data = batch_data.reshape(batch_data.shape[0] * batch_data.shape[1], batch_data.shape[2])[batch_node_pos.flatten()]
    print("batch_data:", batch_data)
    print("flattened data", flattened_data)
    return message_passing(flattened_idxs, flattened_data, batch_cumsum[-1])

def array_graphs_test():
    hidden = 3
    batch_size = 2
    max_nodes = 7
    data_max = 11
    max_edges = 5
    key = jax.random.PRNGKey(2666)
    nodes_per_graph = jax.random.randint(key, shape=(batch_size,), minval=1, maxval=max_nodes)
    edges_per_graph = jax.random.randint(key, shape=(batch_size,), minval=1, maxval=max_edges)
    print("nodes_per_graph:", nodes_per_graph)
    print("edges_per_graph:", edges_per_graph)
    batch_data = []
    batch_idxs = []
    for i, (num_nodes, num_edges) in enumerate(zip(nodes_per_graph, edges_per_graph)):
        key = jax.random.PRNGKey(i)
        data = jnp.pad(jax.random.randint(key, shape=(num_nodes, hidden), minval=0, maxval=data_max), ((0, max_nodes - num_nodes), (0, 0)))
        edges = jnp.pad(jax.random.randint(key, shape=(num_edges, 2), minval=0, maxval=num_nodes), ((0, max_edges - num_edges), (0, 0)), mode='constant', constant_values=-1)
        batch_data.append(data)
        batch_idxs.append(edges)
    batch_data = jnp.array(batch_data)
    batch_idxs = jnp.array(batch_idxs)
    assert(batch_data.shape == (batch_size, max_nodes, hidden))
    assert(batch_idxs.shape == (batch_size, max_edges, 2))
            

    batched_result = batch_message_passing(batch_idxs, batch_data)
    unbatched_result = jnp.concatenate([message_passing(idxs, data, num_nodes) for idxs, data, num_nodes in zip(batch_idxs, batch_data, nodes_per_graph)], axis=0)
    print("batched_result:", batched_result)
    print("unbatched_result:", unbatched_result)
    passed = jnp.array_equal(batched_result, unbatched_result)
    print(passed)

array_graphs_test()
