"""
On Embeddings for Numerical Features in Tabular Deep Learning.
TensorFlow/Keras re-implementation.
"""

# TODO: finish tensorflow implementation and check if it works

# __version__ = '0.0.12-tf'

__all__ = [
    'LinearEmbeddings',
    'LinearReLUEmbeddings',
    'PeriodicEmbeddings',
    'PiecewiseLinearEmbeddings',
    'PiecewiseLinearEncoding',
    'compute_bins',
]

from typing import Any, Literal, Optional, Union
import tensorflow as tf
import numpy as np
import math
import warnings

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


import tensorflow as tf

def _check_input_shape(x: tf.Tensor, expected_n_features: int) -> None:
    if len(x.shape) < 1:
        raise ValueError(
            f"The input must have at least one dimension, got shape {x.shape}"
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            f"The last dimension of the input was expected to be {expected_n_features}, got {x.shape[-1]}"
        )



class LinearEmbeddings(tf.keras.layers.Layer):
    """
    Linear embeddings for continuous features.

    Args:
        n_features: Number of continuous features.
        d_embedding: Embedding size.

    Input shape: (..., n_features)
    Output shape: (..., n_features, d_embedding)
    """
    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        lim = 1.0 / tf.math.sqrt(tf.cast(d_embedding, tf.float32)).numpy()
        self.weight = self.add_weight(
            shape=(n_features, d_embedding),
            initializer=tf.keras.initializers.RandomUniform(-lim, lim),
            trainable=True,
            name="weight"
        )
        self.bias = self.add_weight(
            shape=(n_features, d_embedding),
            initializer=tf.keras.initializers.RandomUniform(-lim, lim),
            trainable=True,
            name="bias"
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        _check_input_shape(x, self.n_features)
        x = tf.expand_dims(x, -1)  # (..., n_features, 1)
        return self.bias + self.weight * x



class LinearReLUEmbeddings(tf.keras.layers.Layer):
    """
    Simple non-linear embeddings for continuous features.

    Args:
        n_features: Number of continuous features.
        d_embedding: Embedding size (default: 32).

    Input shape: (..., n_features)
    Output shape: (..., n_features, d_embedding)
    """
    def __init__(self, n_features: int, d_embedding: int = 32):
        super().__init__()
        self.linear = LinearEmbeddings(n_features, d_embedding)
        self.activation = tf.keras.layers.ReLU()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.linear(x)
        return self.activation(x)



class _Periodic(tf.keras.layers.Layer):
    """
    NOTE: THIS MODULE SHOULD NOT BE USED DIRECTLY.

    Technically, this is a linear embedding without bias followed by
    the periodic activations. The scale of the initialization
    (defined by the `sigma` argument) plays an important role.
    """

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')
        self._sigma = sigma
        self.n_features = n_features
        self.k = k
        # Truncated normal initialization within [-bound, bound]
        bound = sigma * 3
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=sigma)
        # We'll clip after initialization to [-bound, bound]
        initial_weight = initializer(shape=(n_features, k))
        initial_weight = tf.clip_by_value(initial_weight, -bound, bound)
        self.weight = tf.Variable(initial_weight, trainable=True, name="weight")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        _check_input_shape(x, self.n_features)
        # x: (..., n_features)
        # self.weight: (n_features, k)
        # x[..., None]: (..., n_features, 1)
        # self.weight[None, ...]: (1, n_features, k) for broadcasting
        # But tf will broadcast automatically
        x_proj = 2 * math.pi * self.weight * tf.expand_dims(x, -1)  # (..., n_features, k)
        x_cos = tf.math.cos(x_proj)
        x_sin = tf.math.sin(x_proj)
        # Concatenate on the last axis (k -> 2k)
        x_out = tf.concat([x_cos, x_sin], axis=-1)
        return x_out


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html

class _NLinear(tf.keras.layers.Layer):
    """N *separate* linear layers for N feature embeddings.

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        d_in_rsqrt = in_features ** -0.5
        # Weight shape: (n, in_features, out_features)
        self.weight = self.add_weight(
            shape=(n, in_features, out_features),
            initializer=tf.keras.initializers.RandomUniform(-d_in_rsqrt, d_in_rsqrt),
            trainable=True,
            name="weight"
        )
        if bias:
            self.bias = self.add_weight(
                shape=(n, out_features),
                initializer=tf.keras.initializers.RandomUniform(-d_in_rsqrt, d_in_rsqrt),
                trainable=True,
                name="bias"
            )
        else:
            self.bias = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, n, in_features)
        if x.ndim != 3:
            raise ValueError(
                '_NLinear supports only inputs with exactly one batch dimension, so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).'
            )
        if x.shape[1] != self.n or x.shape[2] != self.in_features:
            raise ValueError(f"Input shape {x.shape} does not match expected (batch_size, {self.n}, {self.in_features})")
        # For each feature (axis 1), apply its own linear layer to the last axis
        # Output: (batch_size, n, out_features)
        # Use einsum for batch matrix multiplication: 'bni,nio->bno'
        out = tf.einsum('bni,nio->bno', x, self.weight)
        if self.bias is not None:
            out = out + self.bias  # broadcast (n, out_features) over batch
        return out



class PeriodicEmbeddings(tf.keras.layers.Layer):
    """Embeddings for continuous features based on periodic activations.

    See README for details.

    Input shape: (..., n_features)
    Output shape: (..., n_features, d_embedding)
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_frequencies = n_frequencies
        self.frequency_init_scale = frequency_init_scale
        self.activation_flag = activation
        self.lite = lite
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        if lite:
            if not activation:
                raise ValueError('lite=True is allowed only when activation=True')
            # Shared linear layer for all features
            self.linear = tf.keras.layers.Dense(
                d_embedding,
                use_bias=True,
                input_shape=(2 * n_frequencies,),
                name="periodic_dense_lite"
            )
        else:
            # Separate linear layer for each feature
            self.linear = _NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = tf.keras.layers.ReLU() if activation else None

    def get_output_shape(self):
        """Get the output shape without the batch dimensions."""
        return (self.n_features, self.d_embedding)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (..., n_features)
        x = self.periodic(x)  # (..., n_features, 2 * n_frequencies)
        if self.lite:
            # Apply the same Dense layer to each feature vector
            # x: (..., n_features, 2 * n_frequencies)
            # We need to apply Dense to the last axis for each feature
            # Reshape to (-1, 2 * n_frequencies), apply Dense, then reshape back
            orig_shape = tf.shape(x)
            batch_dims = orig_shape[:-2]
            n_features = orig_shape[-2]
            x_flat = tf.reshape(x, [-1, 2 * self.n_frequencies])
            x_out = self.linear(x_flat)
            x_out = tf.reshape(x_out, tf.concat([batch_dims, [n_features, self.d_embedding]], axis=0))
        else:
            # _NLinear expects (batch_size, n, in_features)
            # If input is (..., n_features, 2 * n_frequencies), flatten batch dims
            x_shape = tf.shape(x)
            batch_size = tf.reduce_prod(x_shape[:-2])
            x_reshaped = tf.reshape(x, [batch_size, self.n_features, 2 * self.n_frequencies])
            x_out = self.linear(x_reshaped)
            # Reshape back to original batch dims
            out_shape = tf.concat([x_shape[:-2], [self.n_features, self.d_embedding]], axis=0)
            x_out = tf.reshape(x_out, out_shape)
        if self.activation is not None:
            x_out = self.activation(x_out)
        return x_out



def _check_bins(bins: list) -> None:
    """
    Check that bins is a list of 1D numpy arrays or tf.Tensor, with at least two elements, sorted, and finite.
    """
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        # Accept numpy arrays or tf.Tensor
        if not (isinstance(feature_bins, np.ndarray) or isinstance(feature_bins, tf.Tensor)):
            raise ValueError(
                f'bins must be a list of numpy arrays or tf.Tensor. However, for {i=}: {type(feature_bins)=}'
            )
        arr = feature_bins.numpy() if isinstance(feature_bins, tf.Tensor) else feature_bins
        if arr.ndim != 1:
            raise ValueError(
                f'Each item of the bin list must have exactly one dimension. However, for {i=}: {arr.ndim=}'
            )
        if len(arr) < 2:
            raise ValueError(
                f'All features must have at least two bin edges. However, for {i=}: {len(arr)=}'
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f'Bin edges must not contain nan/inf/-inf. However, this is not true for the {i}-th feature'
            )
        if np.any(arr[:-1] >= arr[1:]):
            raise ValueError(
                f'Bin edges must be sorted. However, for the {i}-th feature, the bin edges are not sorted'
            )
        if len(arr) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
                ' Strictly speaking, using a single bin for the'
                ' piecewise-linear encoding should not break anything,'
                ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
            )



def compute_bins(
    X,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict] = None,
    y: Optional[np.ndarray] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list:
    """Compute the bin boundaries for PiecewiseLinearEncoding and PiecewiseLinearEmbeddings.

    Args:
        X: the training features (numpy array or tf.Tensor, shape (n_samples, n_features)).
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for sklearn.tree.DecisionTreeRegressor or DecisionTreeClassifier.
        y: the training labels (must be provided if tree_kwargs is not None).
        regression: whether the labels are regression labels (must be provided if tree_kwargs is not None).
        verbose: if True and tree_kwargs is not None, tqdm will report progress.

    Returns:
        A list of bin edges for all features (as numpy arrays).
    """
    # Convert tf.Tensor to numpy if needed
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    if not isinstance(X, np.ndarray):
        raise ValueError(f'X must be a numpy array or tf.Tensor, got {type(X)}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, got {X.ndim}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, got {X.shape[0]}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, got {X.shape[1]}')
    if not np.isfinite(X).all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if np.any(np.all(X == X[0], axis=0)):
        raise ValueError('All columns of X must have at least two distinct values.')
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(f'n_bins must be more than 1, but less than len(X), got n_bins={n_bins}, len(X)={len(X)}')

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError('If tree_kwargs is None, then y, regression must be None and verbose must be False')
        # Quantile-based bins
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = [
            np.unique(np.quantile(X[:, i], quantiles))
            for i in range(X.shape[1])
        ]
        _check_bins(bins)
        return bins
    else:
        if sklearn_tree is None:
            raise RuntimeError('The scikit-learn package is missing. See README.md for installation instructions')
        if y is None or regression is None:
            raise ValueError('If tree_kwargs is not None, then y and regression must not be None')
        if isinstance(y, tf.Tensor):
            y = y.numpy()
        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a numpy array or tf.Tensor')
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, got {y.ndim}')
        if len(y) != len(X):
            raise ValueError(f'len(y) must be equal to len(X), got len(y)={len(y)}, len(X)={len(X)}')
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError('tree_kwargs must not contain the key "max_leaf_nodes" (it will be set to n_bins automatically).')

        tqdm_ = tqdm if (verbose and tqdm is not None) else lambda x: x
        bins = []
        for i, column in enumerate(tqdm_(X.T)):
            feature_bin_edges = [float(np.min(column)), float(np.max(column))]
            tree_cls = sklearn_tree.DecisionTreeRegressor if regression else sklearn_tree.DecisionTreeClassifier
            tree = tree_cls(max_leaf_nodes=n_bins, **tree_kwargs).fit(column.reshape(-1, 1), y).tree_
            for node_id in range(tree.node_count):
                # Only for split nodes
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(np.unique(feature_bin_edges))
        _check_bins(bins)
        return bins



class _PiecewiseLinearEncodingImpl(tf.keras.layers.Layer):
    """Piecewise-linear encoding (TensorFlow version).

    NOTE: THIS CLASS SHOULD NOT BE USED DIRECTLY.
    In particular, this class does *not* add any positional information
    to feature encodings. Thus, for Transformer-like models,
    `PiecewiseLinearEmbeddings` is the only valid option.
    """
    def __init__(self, bins: list) -> None:
        super().__init__()
        assert len(bins) > 0
        n_features = len(bins)
        n_bins = [len(np.array(x)) - 1 for x in bins]
        max_n_bins = max(n_bins)

        # Store weight and bias as non-trainable variables (constant)
        weight = np.zeros((n_features, max_n_bins), dtype=np.float32)
        bias = np.zeros((n_features, max_n_bins), dtype=np.float32)

        for i, bin_edges in enumerate(bins):
            bin_edges = np.array(bin_edges)
            bin_width = np.diff(bin_edges)
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            weight[i, -1] = w[-1]
            bias[i, -1] = b[-1]
            if n_bins[i] > 1:
                weight[i, : n_bins[i] - 1] = w[:-1]
                bias[i, : n_bins[i] - 1] = b[:-1]
        self.weight = tf.constant(weight, dtype=tf.float32)
        self.bias = tf.constant(bias, dtype=tf.float32)

        # single_bin_mask: shape (n_features,)
        single_bin_mask = np.array(n_bins) == 1
        self.single_bin_mask = tf.constant(single_bin_mask, dtype=tf.bool) if np.any(single_bin_mask) else None

        # mask: shape (n_features, max_n_bins), True for valid (non-padding) components
        if all(len(x) == len(bins[0]) for x in bins):
            self.mask = None
        else:
            mask = []
            for x in bins:
                n = len(x) - 1
                m = max_n_bins
                mask_row = np.concatenate([
                    np.ones(n - 1, dtype=bool) if n > 1 else np.zeros(0, dtype=bool),
                    np.zeros(m - n, dtype=bool),
                    np.ones(1, dtype=bool)
                ])
                mask.append(mask_row)
            self.mask = tf.constant(np.stack(mask, axis=0), dtype=tf.bool)

    def get_max_n_bins(self) -> int:
        return int(self.weight.shape[-1])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (..., n_features)
        # self.weight: (n_features, max_n_bins)
        # self.bias: (n_features, max_n_bins)
        x = tf.expand_dims(x, -1)  # (..., n_features, 1)
        out = self.bias + self.weight * x  # (..., n_features, max_n_bins)
        # Now apply clamping logic
        if out.shape[-1] > 1:
            # Clamp first, middle, and last components as in the original
            first = tf.clip_by_value(out[..., :1], clip_value_min=-np.inf, clip_value_max=1.0)
            middle = tf.clip_by_value(out[..., 1:-1], 0.0, 1.0)
            last = out[..., -1:]
            if self.single_bin_mask is None:
                last = tf.clip_by_value(last, 0.0, np.inf)
            else:
                # For features with only one bin, do not clamp last component
                mask = tf.cast(self.single_bin_mask, out.dtype)
                last = mask * last + (1.0 - mask) * tf.clip_by_value(last, 0.0, np.inf)
                # Broadcast mask to match batch dims
                for _ in range(len(out.shape) - 2):
                    last = tf.expand_dims(last, 0)
            out = tf.concat([first, middle, last], axis=-1)
        return out



class PiecewiseLinearEncoding(tf.keras.layers.Layer):
    """Piecewise-linear encoding (TensorFlow version).

    See README for detailed explanation.

    Input: (..., n_features)
    Output: (..., total_n_bins), where total_n_bins = sum(len(b) - 1 for b in bins)
    """
    def __init__(self, bins: list) -> None:
        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

        # Compute total_n_bins for output shape
        if self.impl.mask is None:
            self._total_n_bins = int(np.prod(self.impl.weight.shape))
        else:
            self._total_n_bins = int(np.sum(self.impl.mask.numpy().astype(np.int32)))

    def get_output_shape(self):
        """Get the output shape without the batch dimensions."""
        return (self._total_n_bins,)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (..., n_features)
        x = self.impl(x)
        if self.impl.mask is None:
            # Flatten last two dims
            shape = tf.shape(x)
            new_shape = tf.concat([shape[:-2], [shape[-2] * shape[-1]]], axis=0)
            return tf.reshape(x, new_shape)
        else:
            # Mask out padding (keep only valid components)
            # x: (..., n_features, max_n_bins), mask: (n_features, max_n_bins)
            # We'll flatten the last two dims and apply the mask
            mask = self.impl.mask
            x_flat = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], axis=0))
            mask_flat = tf.reshape(mask, [-1])
            # Only keep the masked (True) elements for each sample
            # tf.boolean_mask applies mask to the last dim
            return tf.boolean_mask(x_flat, mask_flat, axis=-1)



class PiecewiseLinearEmbeddings(tf.keras.layers.Layer):
    """Piecewise-linear embeddings (TensorFlow version).

    Input: (batch_size, n_features)
    Output: (batch_size, n_features, d_embedding)
    """
    def __init__(
        self,
        bins: list,
        d_embedding: int,
        *,
        activation: bool,
        version: Optional[str] = None,
    ):
        super().__init__()
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be a positive integer, got {d_embedding}')
        _check_bins(bins)
        if version is None:
            warnings.warn(
                'The `version` argument is not provided, so version="A" will be used'
                ' for backward compatibility.'
                ' See README for recommendations regarding `version`.'
                ' In future, omitting this argument will result in an exception.'
            )
            version = 'A'
        n_features = len(bins)
        is_version_B = version == 'B'
        self.linear0 = LinearEmbeddings(n_features, d_embedding) if is_version_B else None
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(
            n_features,
            self.impl.get_max_n_bins(),
            d_embedding,
            bias=not is_version_B,
        )
        # In PyTorch version, version B zero-initializes self.linear.weight
        # In TensorFlow, we cannot directly zero the weights after creation, but can set initializer if needed
        self.activation = tf.keras.layers.ReLU() if activation else None

    def get_output_shape(self):
        n_features = self.linear.n
        d_embedding = self.linear.out_features
        return (n_features, d_embedding)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, n_features)
        if x.ndim != 2:
            raise ValueError('For now, only inputs with exactly one batch dimension are supported.')
        x_linear = self.linear0(x) if self.linear0 is not None else None
        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)
        if self.activation is not None:
            x_ple = self.activation(x_ple)
        return x_ple if x_linear is None else x_linear + x_ple
