import numpy as np

# Index into ELEMENT_MAP is atomic number, value is type number. -99 indicates element not in dataset.
ELEMENT_MAP = onp.array([ 0,  1, -99,  2, -99, -99,  3,  4,  5,  6, -99,  7,  8, -99, -99,  9, 10, 11, -99, 12, 13, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 14, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, -99, 15]) 

NUM_ELEMENTS = 16

def select(subset_labels, subset, *fields):
    selection = (subset_labels == subset)
    return (field[selection] for field in *fields)

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
    print(pos)
    assert(len(pos.shape) == 2)
    return np.argwhere((distance_matrix(pos) < L) & ~np.identity(pos.shape[-2], dtype=bool))

"""
Batched geometry (batch_size, n_atoms, 3) -> indices of edges within L (batch_size, n_edges, 2)
"""
def batch_radius_graph(batch_pos, L, max_edges):
    assert(len(batch_pos.shape) == 3)
    all_edges = []    
    for i, pos in enumerate(batch_pos):
        edges = radius_graph(pos, L)
        pad_len = max_edges - len(edges)
        if pad_len < 0:
            print(f"Skipping {i}")
            continue
        all_edges.append(np.pad(edges, ((0, pad_len), (0, 0)), mode="constant", constant_values=-1))
    return np.array(all_edges, dtype=np.int8)
