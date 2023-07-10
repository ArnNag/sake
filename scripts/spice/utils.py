import numpy as np

"""
Geometry (n_atoms, 3) -> pairwise distances (n_atoms, n_atoms)
Batched geometry (batch_size, n_atoms, 3) -> pairwise distances (batch_size, n_atoms, n_atoms)
"""
def distance_matrix(pos):
    assert(len(pos.shape) == 2 or len(pos.shape) == 3)
    return np.linalg.norm(np.expand_dims(pos, -2) - np.expand_dims(pos, -3), axis=-1)


"""
Geometry (n_atoms, 3) -> indices of edges within L (n_edges, 2)
"""
def radius_graph(pos, L):
    assert(len(pos.shape) == 2)
    return np.argwhere((distance_matrix(pos) < L) & ~np.identity(pos.shape[-2], dtype=bool))

"""
Batched geometry (batch_size, n_atoms, 3) -> pairwise distances (batch_size, n_atoms, n_atoms)
"""
def batch_radius_graph(batch_pos, L, max_edges):
    assert(len(pos.shape) == 3)
    all_edges = []    
    for i, pos in enumerate(batch_pos):
        edges = radius_graph(pos, L)
        pad_len = max_edges - len(edges)
        if pad_len < 0:
            print(f"Skipping {i}")
            continue
        all_edges.append(np.pad(edges, ((0, pad_len), (0, 0)), mode="constant", constant_values=-1))
    return np.array(all_edges)

# """
# Geometry (batch_size, n_atoms, 3) -> indices of edges within L (batch_size, n_edges 2)
# """
# def batch_radius_graph(pos, cutoff):
#     _edges = radius_graph(pos, cutoff)
#     return np.split(_edges[:,1:], np.where(np.diff(_edges[:,0]))[0] + 1)






