# rtdl_num_embeddings_keras.py
# Keras 3 (keras-core) implementation of numeric embedding layers
# Inspired by: https://github.com/yandex-research/rtdl-num-embeddings
# Implementation by Linnea Wolniewicz, 2025 — ported to Keras 3 by ChatGPT

from __future__ import annotations

from typing import Optional, Literal, Any
import math

import keras_core as keras
from keras_core import ops


def _check_input_shape(x: Any, n_features: int, name: str = "input"):
    """Best-effort static check; skips when shape is dynamic."""
    try:
        last = x.shape[-1]
        if last is not None and int(last) != int(n_features):
            raise ValueError(
                f"{name} last dimension must be n_features={n_features}, got shape {x.shape}."
            )
    except Exception:
        # Shape unknown at trace-time; let the backend validate later.
        pass


class _ClipConstraint(keras.constraints.Constraint):
    """Clips weights elementwise to [min_value, max_value]."""
    def __init__(self, min_value: float, max_value: float):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        return ops.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


@keras.saving.register_keras_serializable(package="rtdl")
class LinearEmbeddings(keras.layers.Layer):
    """
    Linear embeddings for continuous features.
    Output: (batch, n_features, d_embedding)
    """
    def __init__(self, n_features: int, d_embedding: int, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if d_embedding <= 0:
            raise ValueError("d_embedding must be positive")
        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)

    def build(self, input_shape):
        d_rsqrt = self.d_embedding ** -0.5
        init = keras.initializers.RandomUniform(minval=-d_rsqrt, maxval=d_rsqrt)
        self.weight = self.add_weight(
            name="weight",
            shape=(self.n_features, self.d_embedding),
            initializer=init,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_features, self.d_embedding),
            initializer=init,
            trainable=True,
        )

    def call(self, x):
        _check_input_shape(x, self.n_features)
        x_exp = ops.expand_dims(x, axis=-1)          # (B, F, 1)
        out = self.bias + self.weight * x_exp        # (B, F, D)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "d_embedding": self.d_embedding})
        return cfg


@keras.saving.register_keras_serializable(package="rtdl")
class LinearReLUEmbeddings(keras.layers.Layer):
    """ReLU(LinearEmbeddings(...))."""
    def __init__(self, n_features: int, d_embedding: int, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)
        self.linear = LinearEmbeddings(n_features, d_embedding)
        self.relu = keras.layers.ReLU()

    def build(self, input_shape):
        # Build sublayers with known static shapes
        self.linear.build(input_shape)  # (B, F) -> (B, F, d_embed)
        out_shape = tuple(input_shape[:-1]) + (self.n_features, self.d_embedding)
        self.relu.build(out_shape)
        super().build(input_shape)

    def call(self, x):
        return self.relu(self.linear(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "d_embedding": self.d_embedding})
        return cfg


@keras.saving.register_keras_serializable(package="rtdl")
class _Periodic(keras.layers.Layer):
    """Linear transform (no bias) -> concat(cos, sin)."""
    def __init__(self, n_features: int, k: int, sigma: float, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.n_features = int(n_features)
        self.k = int(k)
        self.sigma = float(sigma)

    def build(self, input_shape):
        bound = self.sigma * 3.0
        self.weight = self.add_weight(
            name="weight",
            shape=(self.n_features, self.k),
            initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=self.sigma),
            trainable=True,
            constraint=_ClipConstraint(-bound, bound),  # keep weights within ±3σ
        )

    def call(self, x):
        _check_input_shape(x, self.n_features)
        arg = 2.0 * math.pi * ops.expand_dims(x, -1) * self.weight  # (B, F, K)
        return ops.concatenate([ops.cos(arg), ops.sin(arg)], axis=-1)  # (B, F, 2K)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "k": self.k, "sigma": self.sigma})
        return cfg


@keras.saving.register_keras_serializable(package="rtdl")
class _NLinear(keras.layers.Layer):
    """N separate linear layers: (B, N, Din) -> (B, N, Dout)."""
    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True,
                 name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n = int(n)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)

    def build(self, input_shape):
        limit = (self.in_features ** -0.5)
        init = keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        self.weight = self.add_weight(
            name="weight",
            shape=(self.n, self.in_features, self.out_features),
            initializer=init,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.n, self.out_features),
                initializer=init,
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, x):
        # x: (B, N, Din), weight: (N, Din, Dout)
        if len(x.shape) != 3:
            raise ValueError("x must be of shape (batch, n, d_in)")
        y = ops.einsum("bnd,ndo->bno", x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n": self.n,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.use_bias,
        })
        return cfg


@keras.saving.register_keras_serializable(package="rtdl")
class PeriodicEmbeddings(keras.layers.Layer):
    """
    Periodic embeddings: _Periodic -> _NLinear (+ optional residual) -> ReLU (optional)
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        *,
        k: int = 64,
        sigma: float = 0.02,
        activation: bool = True,
        version: Optional[Literal["A", "B"]] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if n_features <= 0 or d_embedding <= 0:
            raise ValueError("n_features and d_embedding must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")

        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)
        self.k = int(k)
        self.sigma = float(sigma)
        self.version = version

        self.periodic = _Periodic(n_features, k, sigma)
        self.linear = _NLinear(n_features, 2 * k, d_embedding, bias=True)
        self.activation = keras.layers.ReLU() if activation else None
        self.linear0 = LinearEmbeddings(n_features, d_embedding) if version == "B" else None

    def build(self, input_shape):
        # Input: (B, F)
        self.periodic.build(input_shape)  # -> (B, F, 2k)
        periodic_out = tuple(input_shape[:-1]) + (self.n_features, 2*self.k)
        self.linear.build(periodic_out)   # -> (B, F, d_embedding)
        if self.linear0 is not None:
            self.linear0.build(input_shape)
        if self.activation is not None:
            act_in = tuple(input_shape[:-1]) + (self.n_features, self.d_embedding)
            self.activation.build(act_in)
        super().build(input_shape)

    def call(self, x):
        _check_input_shape(x, self.n_features)
        x_res = self.linear0(x) if self.linear0 is not None else None
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x if x_res is None else (x + x_res)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_features": self.n_features,
            "d_embedding": self.d_embedding,
            "k": self.k,
            "sigma": self.sigma,
            "activation": self.activation is not None,
            "version": self.version,
        })
        return cfg

# TODO: Something seems to be wrong with the PiecewiseLinearEmbeddings, loss is nan when using it. Check against PyTorch version.
# Also, not set up for keras_core yet, instead tf.keras is used.
# def _prepare_bins(bins: Sequence[np.ndarray | tf.Tensor]):
#     bins_tf: List[tf.Tensor] = []
#     n_bins_per_feature: List[int] = []
#     for i, b in enumerate(bins):
#         bt = tf.convert_to_tensor(b, dtype=tf.float32)
#         if bt.shape.rank != 1 or bt.shape[0] < 2:
#             raise ValueError(f"Each bins[{i}] must be 1D with at least two edges")
#         bins_tf.append(bt)
#         n_bins_per_feature.append(int(bt.shape[0]) - 1)
#     max_n_bins = int(max(n_bins_per_feature))
#     if any(nb != max_n_bins for nb in n_bins_per_feature):
#         rows = []
#         for nb in n_bins_per_feature:
#             keep = np.concatenate([
#                 np.ones(nb - 1, dtype=bool),
#                 np.zeros(max_n_bins - nb, dtype=bool),
#                 np.ones(1, dtype=bool),
#             ])
#             rows.append(keep)
#         mask = tf.convert_to_tensor(np.concatenate(rows, axis=0))
#     else:
#         mask = None
#     return bins_tf, max_n_bins, mask

# class _PiecewiseLinearEncodingImpl(tf.keras.layers.Layer):
#     def __init__(self, bins: Sequence[np.ndarray | tf.Tensor], name: str | None = None):
#         super().__init__(name=name)
#         self._bins_raw = bins
#         self._built_constants = False
#     def build(self, input_shape):
#         bins_tf, max_n_bins, mask = _prepare_bins(self._bins_raw)
#         self.n_features = len(bins_tf)
#         self.max_n_bins = max_n_bins
#         self.mask = mask
#         edges = tf.ragged.stack(bins_tf).to_tensor(shape=(self.n_features, max_n_bins + 1), default_value=np.nan)
#         self.bin_left = self.add_weight(
#             name="bin_left", shape=edges[:, :-1].shape, initializer=tf.keras.initializers.Constant(edges[:, :-1]), trainable=False
#         )
#         self.bin_right = self.add_weight(
#             name="bin_right", shape=edges[:, 1:].shape, initializer=tf.keras.initializers.Constant(edges[:, 1:]), trainable=False
#         )
#         self._built_constants = True
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         _check_input_shape(x, int(self.n_features))
#         M = self.max_n_bins
#         x3 = tf.expand_dims(x, -1)  # (B, F, 1)
#         left = self.bin_left       # (F, M)
#         right = self.bin_right     # (F, M)
#         width = right - left
#         width_safe = tf.where(tf.math.is_finite(width), width, tf.ones_like(width))
#         r = (x3 - left) / width_safe  # (B, F, M)
#         inner = tf.clip_by_value(r, 0.0, 1.0)  # (B, F, M)
#         last_bin = tf.nn.relu(x3[..., -1:] - left[..., -1:])  # (B, F, 1)
#         inner_except_last = inner[..., :-1] if M > 1 else tf.zeros_like(inner[..., :0])
#         enc = tf.concat([inner_except_last, last_bin], axis=-1)  # (B, F, M)
#         return enc

# class PiecewiseLinearEncoding(tf.keras.layers.Layer):
#     def __init__(self, bins: Sequence[np.ndarray | tf.Tensor], name: str | None = None):
#         super().__init__(name=name)
#         self.impl = _PiecewiseLinearEncodingImpl(bins)
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         enc = self.impl(x)  # (B, F, M)
#         flat = tf.reshape(enc, [tf.shape(enc)[0], -1])  # (B, F*M)
#         if self.impl.mask is None:
#             return flat
#         else:
#             return tf.boolean_mask(flat, self.impl.mask, axis=1)

# class PiecewiseLinearEmbeddings(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         bins: Sequence[np.ndarray | tf.Tensor],
#         d_embedding: int,
#         *,
#         activation: bool = True,
#         version: Optional[Literal["A", "B"]] = None,
#         name: str | None = None,
#     ):
#         super().__init__(name=name)
#         if d_embedding <= 0:
#             raise ValueError("d_embedding must be positive")
#         self.activation = tf.keras.layers.ReLU() if activation else None
#         self.impl = _PiecewiseLinearEncodingImpl(bins)
#         self._linear = None
#         self._linear0 = None
#         self._version = version
#         self._d_embedding = int(d_embedding)
#     def build(self, input_shape):
#         if not self.impl._built_constants:
#             dummy = tf.zeros((1, int(input_shape[-1])))
#             _ = self.impl(dummy)
#         F = self.impl.n_features
#         M = self.impl.max_n_bins
#         self._linear = _NLinear(F, M, self._d_embedding, bias=True)
#         if self._version == "B":
#             self._linear0 = LinearEmbeddings(F, self._d_embedding)
#         else:
#             self._linear0 = None
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         if x.shape.rank != 2:
#             raise ValueError("For now, only (batch, n_features) inputs are supported.")
#         x_res = self._linear0(x) if self._linear0 is not None else None
#         x_enc = self.impl(x)              # (B, F, M)
#         x_out = self._linear(x_enc)       # (B, F, D)
#         if self.activation is not None:
#             x_out = self.activation(x_out)
#         return x_out if x_res is None else (x_out + x_res)

# def compute_bins(
#     X: np.ndarray | tf.Tensor,
#     n_bins: int = 48,
#     *,
#     tree_kwargs: Optional[dict[str, Any]] = None,
#     y: Optional[np.ndarray | tf.Tensor] = None,
#     regression: Optional[bool] = None,
#     verbose: bool = False,
# ) -> List[tf.Tensor]:
#     """
#     Compute bin boundaries per feature for Piecewise encodings.

#     Modes:
#       1) y is None: quantile-based bins (n_bins equal-frequency bins per feature).
#       2) y is provided: supervised bins via scikit-learn decision tree thresholds
#          with at most `n_bins` leaves.
#     Returns:
#       list of tf.Tensor of shape (n_edges_j,) for each feature.
#     """
#     X_np = X.numpy() if isinstance(X, tf.Tensor) else np.asarray(X)
#     if X_np.ndim != 2:
#         raise ValueError("X must be 2D (n_samples, n_features)")
#     n_samples, n_features = X_np.shape
#     bins: List[tf.Tensor] = []
#     if y is None:
#         qs = np.linspace(0.0, 1.0, n_bins + 1)
#         for j in range(n_features):
#             col = X_np[:, j]
#             edges = np.quantile(col, qs, method="linear")
#             edges = np.unique(edges)
#             if edges.size < 2:
#                 mn = np.min(col)
#                 mx = np.max(col)
#                 edges = np.array([mn, mx], dtype=np.float32)
#             bins.append(tf.convert_to_tensor(edges.astype(np.float32)))
#         return bins
#     try:
#         from sklearn import tree as sklearn_tree  # type: ignore
#     except Exception as e:
#         raise RuntimeError(
#             "scikit-learn is required for supervised binning. "
#             "Install scikit-learn or call compute_bins with y=None."
#         ) from e
#     y_np = y.numpy() if isinstance(y, tf.Tensor) else np.asarray(y)
#     if regression is None:
#         regression = (y_np.dtype.kind in "f")
#     for j in range(n_features):
#         col = X_np[:, j].reshape(-1, 1)
#         if regression:
#             est = sklearn_tree.DecisionTreeRegressor(max_leaf_nodes=n_bins, **(tree_kwargs or {}))
#         else:
#             est = sklearn_tree.DecisionTreeClassifier(max_leaf_nodes=n_bins, **(tree_kwargs or {}))
#         est.fit(col, y_np)
#         tr = est.tree_
#         thr = []
#         for node_id in range(tr.node_count):
#             if tr.children_left[node_id] != tr.children_right[node_id]:
#                 thr.append(float(tr.threshold[node_id]))
#         edges = np.unique(np.asarray(thr, dtype=np.float32))
#         if edges.size == 0:
#             mn = np.min(col)
#             mx = np.max(col)
#             edges = np.array([mn, mx], dtype=np.float32)
#         bins.append(tf.convert_to_tensor(edges.astype(np.float32)))
#     return bins