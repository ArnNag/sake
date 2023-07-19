import jax
import jax.numpy as jnp
from typing import Optional, Union, Iterable, Mapping, Any

EPSILON = 1e-5
INF = 1e5

def get_x_minus_xt(x):
    return jnp.expand_dims(x, -3) - jnp.expand_dims(x, -2)

def get_x_minus_xt_sparse(x, edges):
    # x_edges shape: (n_nodes, 2, 3)
    x_edges = x[edges]
    return x_edges[:,1,:] - x_edges[:,0,:] # shape: (n_nodes, 3)

def get_x_minus_xt_norm(
    x_minus_xt,
    epsilon: float=EPSILON,
):
    x_minus_xt_norm = (
        jax.nn.relu((x_minus_xt ** 2).sum(axis=-1, keepdims=True))
        + epsilon
    ) ** 0.5

    return x_minus_xt_norm

# def get_h_cat_ht(h):
#     n_nodes = h.shape[-2]
#     h_cat_ht = jnp.concatenate(
#         [
#             jnp.repeat(jnp.expand_dims(h, -3), n_nodes, -3),
#             jnp.repeat(jnp.expand_dims(h, -2), n_nodes, -2)
#         ],
#         axis=-1
#     )
#
#     return h_cat_ht

def get_h_cat_ht(h):
    n_nodes = h.shape[-2]
    h_shape = (*h.shape[:-2], n_nodes, n_nodes, h.shape[-1])
    h_cat_ht = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.expand_dims(h, -3), h_shape),
            jnp.broadcast_to(jnp.expand_dims(h, -2), h_shape),
        ],
        axis=-1,
    )

    return h_cat_ht

def get_h_cat_ht_sparse(h, edges):
    return h[edges].reshape(edges.shape[0], -1)

ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]
def segment_softmax(logits: jnp.ndarray,
                    segment_ids: jnp.ndarray,
                    num_segments: Optional[int] = None,
                    indices_are_sorted: bool = False,
                    unique_indices: bool = False) -> ArrayTree:
    """Computes a segment-wise softmax.

    For a given tree of logits that can be divded into segments, computes a
    softmax over the segments.

    logits = jnp.ndarray([1.0, 2.0, 3.0, 1.0, 2.0])
    segment_ids = jnp.ndarray([0, 0, 0, 1, 1])
    segment_softmax(logits, segments)
    >> DeviceArray([0.09003057, 0.24472848, 0.66524094, 0.26894142, 0.7310586],
    >> dtype=float32)

    Args:
    logits: an array of logits to be segment softmaxed.
    segment_ids: an array with integer dtype that indicates the segments of
      `data` (along its leading axis) to be maxed over. Values can be repeated
      and need not be sorted. Values outside of the range [0, num_segments) are
      dropped and do not contribute to the result.
    num_segments: optional, an int with positive value indicating the number of
      segments. The default is ``jnp.maximum(jnp.max(segment_ids) + 1,
      jnp.max(-segment_ids))`` but since ``num_segments`` determines the size of
      the output, a static value must be provided to use ``segment_sum`` in a
      ``jit``-compiled function.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted
    unique_indices: whether ``segment_ids`` is known to be free of duplicates

    Returns:
    The segment softmax-ed ``logits``.
    """
    # First, subtract the segment max for numerical stability
    maxs = segment_max(logits, segment_ids, num_segments, indices_are_sorted,
                     unique_indices)
    logits = logits - maxs[segment_ids]
    # Then take the exp
    logits = jnp.exp(logits)
    # Then calculate the normalizers
    normalizers = segment_sum(logits, segment_ids, num_segments,
                            indices_are_sorted, unique_indices)
    normalizers = normalizers[segment_ids]
    softmax = logits / normalizers
    return softmax

def segment_mean(data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
  """Returns mean for each segment.
  Args:
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  """
  nominator = jax.ops.segment_sum(
      data,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  denominator = jax.ops.segment_sum(
      jnp.ones_like(data),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  return nominator / jnp.maximum(denominator,
                                 jnp.ones(shape=[], dtype=denominator.dtype))

