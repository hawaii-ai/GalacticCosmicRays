"""
Keras Core (Keras 3) implementations of two numeric-embedding layers inspired by
"On Embeddings for Numerical Features in Tabular Deep Learning" (NeurIPS 2022):
- PeriodicEmbeddings
- PiecewiseLinearEncoding

Design goals
------------
• Backend-agnostic (TensorFlow / JAX / PyTorch backends via Keras Core)
• Feature-wise parameters
• Careful handling of shapes, dtypes, and NaNs
• Optional data-driven calibration via .adapt() where appropriate

Author: ChatGPT (Keras Core)
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import math
import keras_core as keras
from keras_core import ops


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _check_last_dim(x, n_features: int, name: str = "input"):
    if x.shape[-1] != n_features:
        raise ValueError(
            f"{name} last dimension must be n_features={n_features}, got {x.shape}."
        )


def _safe_minmax(x, axis=0):
    """Return (min, max) with safe handling for degenerate ranges."""
    x_min = ops.min(x, axis=axis)
    x_max = ops.max(x, axis=axis)
    return x_min, x_max

# ---------------------------------------------------------------------------
# PeriodicEmbeddings
# ---------------------------------------------------------------------------
class PeriodicEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        learnable_frequencies: bool = True,
        learnable_phases: bool = False,
        frequency_init: str = "loglinear",
        w_min: float = 1.0,
        w_max: float = 1000.0,
        use_phase: bool = False,
        concatenate_sin_cos: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_features = int(n_features)
        self.n_frequencies = int(n_frequencies)
        self.learnable_frequencies = bool(learnable_frequencies)
        self.learnable_phases = bool(learnable_phases)
        self.frequency_init = frequency_init
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.use_phase = bool(use_phase)
        self.concatenate_sin_cos = bool(concatenate_sin_cos)

    def build(self, input_shape):
        weight_dtype = getattr(getattr(self, "dtype_policy", None), "variable_dtype", "float32")
        if self.frequency_init == "loglinear":
            base = ops.linspace(0.0, 1.0, self.n_frequencies)
            base = ops.cast(base, weight_dtype)
            ratio_t = ops.cast(ops.convert_to_tensor(self.w_max / self.w_min), weight_dtype)
            wmin_t = ops.cast(ops.convert_to_tensor(self.w_min), weight_dtype)
            ws = wmin_t * ops.exp(base * ops.log(ratio_t))
            w_init = ops.reshape(ws, (1, self.n_frequencies))
            w_init = ops.repeat(w_init, self.n_features, axis=0)
        elif self.frequency_init == "normal":
            w_init = keras.random.normal((self.n_features, self.n_frequencies), stddev=1.0, dtype=weight_dtype)
            w_init = ops.abs(w_init) + ops.cast(1e-3, weight_dtype)
        else:
            w_init = keras.random.uniform(
                (self.n_features, self.n_frequencies),
                minval=self.w_min,
                maxval=self.w_max,
                dtype=weight_dtype,
            )
        w_init = ops.cast(w_init, weight_dtype)
        self.w = self.add_weight(
            name="frequencies",
            shape=(self.n_features, self.n_frequencies),
            dtype=weight_dtype,
            initializer=keras.initializers.Constant(ops.convert_to_tensor(w_init, dtype=weight_dtype)),
            trainable=self.learnable_frequencies,
        )
        if self.use_phase:
            self.phi = self.add_weight(
                name="phases",
                shape=(self.n_features, self.n_frequencies),
                dtype=weight_dtype,
                initializer="zeros",
                trainable=self.learnable_phases,
            )
        else:
            self.phi = None

    def call(self, x):
        _check_last_dim(x, self.n_features)
        x = ops.convert_to_tensor(x)
        x_exp = ops.expand_dims(x, axis=-1)
        w = ops.cast(self.w, x.dtype)
        arg = x_exp * w
        if self.phi is not None:
            arg = arg + ops.cast(self.phi, x.dtype)
        sin_part = ops.sin(arg)
        if self.concatenate_sin_cos:
            cos_part = ops.cos(arg)
            out = ops.concatenate([sin_part, cos_part], axis=-1)
        else:
            out = sin_part
        out = ops.reshape(out, (-1, self.n_features * self.n_frequencies * (2 if self.concatenate_sin_cos else 1)))
        return out

    def get_config(self):
        base = super().get_config()
        base.update({
            "n_features": self.n_features,
            "n_frequencies": self.n_frequencies,
            "learnable_frequencies": self.learnable_frequencies,
            "learnable_phases": self.learnable_phases,
            "frequency_init": self.frequency_init,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "use_phase": self.use_phase,
            "concatenate_sin_cos": self.concatenate_sin_cos,
        })
        return base

# ---------------------------------------------------------------------------
# PiecewiseLinearEncoding (fixed, non-trainable). Outputs segment weights.
# ---------------------------------------------------------------------------
class PiecewiseLinearEncoding(keras.layers.Layer):
    """Fixed piecewise-linear encoding (PLE) per the RTDL spec.

    For each feature value x, outputs a length-`n_bins` vector of segment
    weights: ones for all bins strictly before the current bin, the fractional
    part in the current bin, zeros afterwards. Per-feature encodings are
    concatenated -> shape (batch, n_features * n_bins).

    This layer has **no trainable embeddings**; only min/max calibration.
    """
    def __init__(
        self,
        n_features: int,
        n_bins: int,
        clip: bool = True,
        use_adaptive_range: bool = True,
        value_range: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if n_features <= 0 or n_bins <= 0:
            raise ValueError("n_features and n_bins must be positive.")
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.clip = bool(clip)
        self.use_adaptive_range = bool(use_adaptive_range)
        self._has_range = False
        self._pending_range = value_range

    def build(self, input_shape):
        weight_dtype = getattr(getattr(self, "dtype_policy", None), "variable_dtype", "float32")
        # Stored mins/maxs (non-trainable, serialized)
        self._mins_var = self.add_weight(
            name="mins", shape=(self.n_features,), dtype=weight_dtype, initializer="zeros", trainable=False
        )
        self._maxs_var = self.add_weight(
            name="maxs", shape=(self.n_features,), dtype=weight_dtype, initializer="ones", trainable=False
        )
        # Sentinel: 1.0 iff adapt() or fixed value_range was set
        self._range_ready = self.add_weight(
            name="range_ready", shape=(), dtype=weight_dtype, initializer="zeros", trainable=False
        )
        if not self.use_adaptive_range:
            if self._pending_range is None:
                raise ValueError("Must provide value_range when use_adaptive_range=False.")
            mins, maxs = self._pending_range
            self._mins_var.assign(ops.convert_to_tensor(mins, dtype=self._mins_var.dtype))
            self._maxs_var.assign(ops.convert_to_tensor(maxs, dtype=self._maxs_var.dtype))
            self._has_range = True
            self._range_ready.assign(ops.cast(1.0, self._range_ready.dtype))

    def adapt(self, data):
        if not getattr(self, "built", False):
            self.build((None, self.n_features))
        x = ops.convert_to_tensor(data)
        _check_last_dim(x, self.n_features, name="adapt data")
        mins = ops.min(x, axis=0)
        maxs = ops.max(x, axis=0)
        eps = ops.cast(1e-6, mins.dtype)
        maxs = ops.where(ops.equal(maxs, mins), maxs + eps, maxs)
        self._mins_var.assign(mins)
        self._maxs_var.assign(maxs)
        self._has_range = True
        self._range_ready.assign(ops.cast(1.0, self._range_ready.dtype))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_features * self.n_bins)

    def call(self, x):
        _check_last_dim(x, self.n_features)
        # During model load, avoid blocking shape inference
        if not self._has_range and self.use_adaptive_range:
            try:
                ready = bool(keras.backend.convert_to_numpy(self._range_ready) > 0.5)
            except Exception:
                ready = False
            if ready:
                self._has_range = True
        x = ops.convert_to_tensor(x)
        x_dtype = x.dtype
        mins = ops.cast(self._mins_var, x_dtype)
        maxs = ops.cast(self._maxs_var, x_dtype)
        ranges = ops.maximum(maxs - mins, ops.cast(1e-6, x_dtype))

        # Normalize to [0, n_bins]; compute i and alpha
        t = (x - mins) / ranges * self.n_bins
        if self.clip:
            t = ops.clip(t, 0.0, float(self.n_bins))
        i = ops.floor(t)
        alpha = t - i
        n_bins_f = ops.cast(self.n_bins, x_dtype)
        i = ops.minimum(i, n_bins_f - ops.cast(1.0, x_dtype))
        alpha = ops.where(t >= n_bins_f, ops.cast(1.0, x_dtype), alpha)
        i0 = ops.cast(i, "int32")

        # Segment weights: ones before i, alpha at i, zeros after
        bin_idx = ops.arange(self.n_bins)
        ones_mask = ops.cast(ops.expand_dims(bin_idx, 0) < ops.expand_dims(i0, -1), x_dtype)
        eq_mask   = ops.cast(ops.expand_dims(bin_idx, 0) == ops.expand_dims(i0, -1), x_dtype)
        seg = ones_mask + eq_mask * ops.expand_dims(alpha, -1)     # (..., n_features, n_bins)

        y = ops.reshape(seg, (-1, self.n_features * self.n_bins))
        return y

    def get_config(self):
        base = super().get_config()
        base.update({
            "n_features": self.n_features,
            "n_bins": self.n_bins,
            "clip": self.clip,
            "use_adaptive_range": self.use_adaptive_range,
            "value_range": None if self._pending_range is None else tuple(self._pending_range),
        })
        return base


# ---------------------------------------------------------------------------
# PiecewiseLinearEmbeddings = Linear(PLE) or ReLU(Linear(PLE)) per feature
# ---------------------------------------------------------------------------
class PiecewiseLinearEmbeddings(keras.layers.Layer):
    """Trainable embeddings on top of PLE, matching RTDL's "Linear(PLE)".

    For each feature, applies a per-feature linear layer (optionally with ReLU)
    to its length-`n_bins` PLE vector to produce a `d_embedding`-dim vector.
    Outputs are concatenated across features -> shape (batch, n_features * d_embedding).
    """
    def __init__(
        self,
        n_features: int,
        n_bins: int,
        d_embedding: int,
        activation: bool = False,  # True -> ReLU(Linear(PLE)), False -> Linear(PLE)
        use_bias: bool = True,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: str | keras.initializers.Initializer = "zeros",
        clip: bool = True,
        use_adaptive_range: bool = True,
        value_range: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if n_features <= 0 or n_bins <= 0 or d_embedding <= 0:
            raise ValueError("n_features, n_bins, d_embedding must be positive.")
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.d_embedding = int(d_embedding)
        self.activation = bool(activation)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.clip = bool(clip)
        self.use_adaptive_range = bool(use_adaptive_range)
        self._pending_range = value_range
        self._has_range = False

    def build(self, input_shape):
        weight_dtype = getattr(getattr(self, "dtype_policy", None), "variable_dtype", "float32")
        # Per-feature kernels: (n_features, n_bins, d_embedding)
        self.kernels = self.add_weight(
            name="kernels",
            shape=(self.n_features, self.n_bins, self.d_embedding),
            dtype=weight_dtype,
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.biases = self.add_weight(
                name="biases",
                shape=(self.n_features, self.d_embedding),
                dtype=weight_dtype,
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.biases = None

        # Range variables (shared behavior with PLE)
        self._mins_var = self.add_weight(
            name="mins", shape=(self.n_features,), dtype=weight_dtype, initializer="zeros", trainable=False
        )
        self._maxs_var = self.add_weight(
            name="maxs", shape=(self.n_features,), dtype=weight_dtype, initializer="ones", trainable=False
        )
        self._range_ready = self.add_weight(
            name="range_ready", shape=(), dtype=weight_dtype, initializer="zeros", trainable=False
        )
        if not self.use_adaptive_range:
            if self._pending_range is None:
                raise ValueError("Must provide value_range when use_adaptive_range=False.")
            mins, maxs = self._pending_range
            self._mins_var.assign(ops.convert_to_tensor(mins, dtype=self._mins_var.dtype))
            self._maxs_var.assign(ops.convert_to_tensor(maxs, dtype=self._maxs_var.dtype))
            self._has_range = True
            self._range_ready.assign(ops.cast(1.0, self._range_ready.dtype))

    def adapt(self, data):
        if not getattr(self, "built", False):
            self.build((None, self.n_features))
        x = ops.convert_to_tensor(data)
        _check_last_dim(x, self.n_features, name="adapt data")
        mins = ops.min(x, axis=0)
        maxs = ops.max(x, axis=0)
        eps = ops.cast(1e-6, mins.dtype)
        maxs = ops.where(ops.equal(maxs, mins), maxs + eps, maxs)
        self._mins_var.assign(mins)
        self._maxs_var.assign(maxs)
        self._has_range = True
        self._range_ready.assign(ops.cast(1.0, self._range_ready.dtype))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_features * self.d_embedding)

    def call(self, x):
        _check_last_dim(x, self.n_features)
        if not self._has_range and self.use_adaptive_range:
            try:
                ready = bool(keras.backend.convert_to_numpy(self._range_ready) > 0.5)
            except Exception:
                ready = False
            if ready:
                self._has_range = True
        x = ops.convert_to_tensor(x)
        x_dtype = x.dtype
        mins = ops.cast(self._mins_var, x_dtype)
        maxs = ops.cast(self._maxs_var, x_dtype)
        ranges = ops.maximum(maxs - mins, ops.cast(1e-6, x_dtype))

        # Normalize to [0, n_bins]; compute i and alpha
        t = (x - mins) / ranges * self.n_bins
        if self.clip:
            t = ops.clip(t, 0.0, float(self.n_bins))
        i = ops.floor(t)
        alpha = t - i
        n_bins_f = ops.cast(self.n_bins, x_dtype)
        i = ops.minimum(i, n_bins_f - ops.cast(1.0, x_dtype))
        alpha = ops.where(t >= n_bins_f, ops.cast(1.0, x_dtype), alpha)
        i0 = ops.cast(i, "int32")

        # Segment weights: ones before i, alpha at i, zeros after
        bin_idx = ops.arange(self.n_bins)
        ones_mask = ops.cast(ops.expand_dims(bin_idx, 0) < ops.expand_dims(i0, -1), x_dtype)
        eq_mask   = ops.cast(ops.expand_dims(bin_idx, 0) == ops.expand_dims(i0, -1), x_dtype)
        seg = ones_mask + eq_mask * ops.expand_dims(alpha, -1)     # (..., n_features, n_bins)

        # Apply per-feature linear: (n_bins) -> (d_embedding), vectorized
        K = ops.cast(self.kernels, x_dtype)          # (F, N, D) = (n_features, n_bins, d_emb)

        # Prepare for batched matmul:
        # seg: (B, F, N) -> (B, F, 1, N)
        # K:   (F, N, D) -> (1, F, N, D)
        seg_exp = ops.expand_dims(seg, axis=-2)      # (B, F, 1, N)
        K_exp   = ops.expand_dims(K,   axis=0)       # (1, F, N, D)

        # Batched matmul over last two dims: (B, F, 1, N) @ (1, F, N, D) -> (B, F, 1, D)
        y = ops.matmul(seg_exp, K_exp)               # (B, F, 1, D)
        y = ops.squeeze(y, axis=-2)                  # (B, F, D)

        # Bias + activation (broadcast over batch)
        if self.use_bias:
            y = y + ops.cast(self.biases, x_dtype)[None, :, :]  # (1, F, D) -> (B, F, D)
        if self.activation:
            y = ops.relu(y)

        # Flatten features
        y = ops.reshape(y, (-1, self.n_features * self.d_embedding))
        return y


    def get_config(self):
        base = super().get_config()
        base.update({
            "n_features": self.n_features,
            "n_bins": self.n_bins,
            "d_embedding": self.d_embedding,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(keras.initializers.get(self.kernel_initializer)),
            "bias_initializer": keras.initializers.serialize(keras.initializers.get(self.bias_initializer)),
            "clip": self.clip,
            "use_adaptive_range": self.use_adaptive_range,
            "value_range": None if self._pending_range is None else tuple(self._pending_range),
        })
        return base


