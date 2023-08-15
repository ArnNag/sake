import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .utils import ExpNormalSmearing
from .functional import get_x_minus_xt, get_x_minus_xt_norm, get_h_cat_ht
from functools import partial
import jraph
from flax.core import frozen_dict


def double_sigmoid(x):
    return 2.0 * jax.nn.sigmoid(x)


class ContinuousFilterConvolutionWithConcatenation(nn.Module):
    out_features: int
    kernel_features: int = 50
    activation: Callable = jax.nn.silu

    def setup(self):
        self.kernel = ExpNormalSmearing(num_rbf=self.kernel_features)
        self.mlp_in = nn.Dense(self.kernel_features)
        self.mlp_out = nn.Sequential(
            [
                nn.Dense(self.out_features),
                self.activation,
                nn.Dense(self.out_features),
            ]
        )

    def __call__(self, h, x):
        h0 = h
        h = self.mlp_in(h)
        _x = self.kernel(x) * h

        h = self.mlp_out(
            jnp.concatenate(
                [h0, _x, x],
                axis=-1
            )
        )

        return h


class SAKELayer(nn.Module):
    out_features: int
    hidden_features: int
    activation: Callable = jax.nn.silu
    n_heads: int = 4
    update: bool = True
    use_semantic_attention: bool = True
    use_euclidean_attention: bool = True
    use_spatial_attention: bool = True
    cutoff: Callable = None

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            [
                # nn.LayerNorm(),
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        if self.update:
            self.velocity_mlp = nn.Sequential(
                [
                    nn.Dense(self.hidden_features),
                    self.activation,
                    nn.Dense(1, use_bias=False),
                    double_sigmoid,
                ],
            )

        self.semantic_attention_mlp = nn.Sequential(
            [
                nn.Dense(self.n_heads),
                partial(nn.celu, alpha=2.0),
            ],
        )

        self.post_norm_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.hidden_features),
                self.activation,
            ]
        )

        self.v_mixing = nn.Dense(1, use_bias=False)
        self.x_mixing = nn.Sequential([nn.Dense(self.n_coefficients, use_bias=False), jnp.tanh])

        log_gamma = -jnp.log(jnp.linspace(1.0, 5.0, self.n_heads))
        if self.use_semantic_attention and self.use_euclidean_attention:
            self.log_gamma = self.param(
                "log_gamma",
                nn.initializers.constant(log_gamma),
                log_gamma.shape,
            )
        else:
            self.log_gamma = jnp.ones(self.n_heads)

    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, graph):
        # coefficients shape: (n_edges, hidden_features * n_heads)
        coefficients = self.x_mixing(h_e_mtx)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5)  # ** 2

        # e: edge axis; x: position axis, c: coefficient axis
        combinations = jnp.einsum("ex,ec->ecx", x_minus_xt, coefficients)

        # combinations_sum shape: (n_nodes, hidden_features * n_heads, 3)
        combinations_sum = jraph.segment_mean(combinations,
                                              graph.receivers,
                                              num_segments=sum(graph.n_node)
                                              )

        combinations_norm = (combinations_sum ** 2).sum(-1)  # .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations, combinations

    @staticmethod
    def aggregate(h_e_mtx, graph):
        return jax.ops.segment_sum(h_e_mtx, graph.receivers, num_segments=sum(graph.n_node))

    def node_model(self, h, h_e, h_combinations):
        out = jnp.concatenate([
                h,
                h_e,
                h_combinations,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def semantic_attention(self, h_e_mtx, graph):
        # att shape: (n_edges, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)
        jax.debug.print("semantic att before softmax: {}", att)
        # return shape: (n_edges, n_heads)
        semantic_attention = jnp.nan_to_num(jraph.segment_softmax(att, graph.receivers, num_segments=sum(graph.n_node)))
        return semantic_attention

    def combined_attention(self, x_minus_xt_norm, h_e_mtx, graph):
        # semantic_attention shape: (n_edges, n_heads)
        semantic_attention = self.semantic_attention(h_e_mtx, graph)
        jax.debug.print("semantic_attention after softmax: {}", semantic_attention)
        if self.cutoff is not None:
            euclidean_attention = self.cutoff(x_minus_xt_norm)
        else:
            euclidean_attention = 1.0

        # combined_attention shape: (n_edges, n_heads)
        combined_attention = euclidean_attention * semantic_attention
        jax.debug.print("combined_attention before normalization: {}", combined_attention)
        # combined_attention_agg shape: (n_nodes, n_heads)
        combined_attention_agg = jax.ops.segment_sum(combined_attention, graph.receivers, num_segments=sum(graph.n_node))
        jax.debug.print("combined_attention_agg: {}", combined_attention_agg)
        jax.debug.print("combined_attention_agg[edges[:,1]]: {}", combined_attention_agg[graph.receivers])
        combined_attention = jnp.nan_to_num(combined_attention / combined_attention_agg[graph.receivers])
        jax.debug.print("combined_attention after normalization: {}", combined_attention)
        
        return combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            graph: jraph.GraphsTuple,
            ) -> jraph.GraphsTuple:

        # x_minus_xt shape: (n_edges, 3)
        x_minus_xt = get_x_minus_xt(graph)
        # x_minus_xt norm shape: (n_edges, 1)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        # h_cat_ht shape: (n_edges, hidden_features * 2 [concatenated sender and receiver]) 
        h_cat_ht = get_h_cat_ht(graph)

        # h_e_mtx shape: (n_edges, hidden_features)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        # combined_attention shape: (n_edges, n_heads)
        combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx, graph)
        # e: edge axis; f: hidden feature axis; h: head axis
        h_e_att = jnp.einsum("ef,eh->efh", h_e_mtx, combined_attention) 
        h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        # h_e_att shape after reshape: (n_edges,  hidden_features * n_heads)
        h_combinations, delta_v = self.spatial_attention(h_e_att, x_minus_xt, x_minus_xt_norm, graph)

        if not self.use_spatial_attention:
            h_combinations = jnp.zeros_like(h_combinations)
            delta_v = jnp.zeros_like(delta_v)

        h_e = self.aggregate(h_e_att, graph)
        h = self.node_model(graph.nodes['h'], h_e, h_combinations)
        x = graph.nodes['x']
        v = graph.nodes['v']

        if self.update:
            # delta_v shape: (n_edges, hidden_features * n_heads, 3)
            delta_v = jax.ops.segment_sum(self.v_mixing(delta_v.swapaxes(-1, -2)).squeeze(-1), graph.receivers, num_segments=sum(graph.n_node))
            delta_v = delta_v / jnp.expand_dims((jax.ops.segment_sum(jnp.ones_like(graph.receivers), graph.receivers, num_segments=sum(graph.n_node)) + 1e-10), -1)
            # delta_v shape after normalization: (n_edges, 3)

            if v is not None:
                v = self.velocity_model(v, h)
            else:
                v = jnp.zeros_like(graph.nodes['x'])

            v = delta_v + v
            x = x + v

        return graph._replace(nodes=graph.nodes.copy({'h': h, 'x': x, 'v': v}))


class EquivariantGraphConvolutionalLayer(nn.Module):
    out_features: int
    hidden_features: int
    activation: Callable = jax.nn.silu
    update: bool = False
    sigmoid: bool = False

    def setup(self):
        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.scaling_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        self.shifting_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        if self.sigmoid:
            self.edge_model = nn.Sequential(
                [
                    nn.Dense(1, use_bias=False),
                    jax.nn.sigmoid,
                ],
            )

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.expand_dims(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_model(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e):
        out = jnp.concatenate([
                h,
                h_e,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            h,
            x,
            v=None,
            mask=None,
            ):

        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)
        h_e_mtx = jnp.concatenate([h_cat_ht, x_minus_xt_norm], axis=-1)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v


class EquivariantGraphConvolutionalLayerWithSmearing(nn.Module):
    out_features: int
    hidden_features: int
    activation: Callable = jax.nn.silu
    update: bool = False
    sigmoid: bool = True

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.scaling_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        self.shifting_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        if self.sigmoid:
            self.edge_att = nn.Sequential(
                [
                    nn.Dense(1, use_bias=False),
                    jax.nn.sigmoid,
                ],
            )

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.expand_dims(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_att(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e):
        out = jnp.concatenate([
                h,
                h_e,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            h,
            x,
            v=None,
            mask=None,
            ):

        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v
