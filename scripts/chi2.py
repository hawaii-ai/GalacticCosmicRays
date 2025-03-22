import numpy as np
import jax.numpy as jnp
from jax.lax import cond, scan

class CalculateChi2:
    """
    Compute interpolated and integrated y between rigidity bins (mLogR).

    Usage:
        mLogR = your mLogR data as a jax numpy array, in logspace
        mLogF = your mLogF data as a jax numpy array, in logspace
        model = ModelChi2(mLogR, mLogF)
        result = model.compute_integral(x1, x2)
    """

    def __init__(self, mLogR, mLogF):
        self.mLogR = mLogR
        self.mLogF = mLogF

    def interpolate_model(self, x):
        lnx = jnp.log(x)
        ig = self.find_grid_point(lnx)

        mLogR_ig = jnp.take(self.mLogR, ig)
        mLogR_ig1 = jnp.take(self.mLogR, ig+1)
        mLogF_ig = jnp.take(self.mLogF, ig)
        mLogF_ig1 = jnp.take(self.mLogF, ig+1)

        # s = (lnx - self.mLogR[ig]) / (self.mLogR[ig+1] - self.mLogR[ig])
        # result = jnp.exp((1 - s) * self.mLogF[ig] + s * self.mLogF[ig+1])

        s = (lnx - mLogR_ig) / (mLogR_ig1 - mLogR_ig)
        result = jnp.exp((1 - s) * mLogF_ig + s * mLogF_ig1)

        return result

    def compute_integral(self, x1, x2):
        lnx1 = jnp.log(x1)
        lnx2 = jnp.log(x2)
        ig1 = self.find_grid_point(lnx1)
        ig2 = self.find_grid_point(lnx2)

        # Compute integral from a to b at grid point ig
        def integral(a, b, ig):
            # M = (self.mLogF[ig+1] - self.mLogF[ig]) / (self.mLogR[ig+1] - self.mLogR[ig])
            # N = jnp.exp(self.mLogF[ig] - M * self.mLogR[ig])

            mLogF_ig = jnp.take(self.mLogF, ig)
            mLogF_ig1 = jnp.take(self.mLogF, ig+1)
            mLogR_ig = jnp.take(self.mLogR, ig)
            mLogR_ig1 = jnp.take(self.mLogR, ig+1)

            M = (mLogF_ig1 - mLogF_ig) / (mLogR_ig1 - mLogR_ig)
            N = jnp.exp(mLogF_ig - M * mLogR_ig)

            Mp1 = M + 1
            N_mp1 = N / Mp1
            mp1b_mp1a = jnp.exp(Mp1 * b) - jnp.exp(Mp1 * a)
            result = N_mp1 * mp1b_mp1a

            return result

        # Compute integral from lnx1 to min(lnx2, self.mLogR[ig1+1]) at grid point ig1
        # I = integral(lnx1, min(lnx2, self.mLogR[ig1+1]), ig1)
        I = integral(lnx1, jnp.minimum(lnx2, jnp.take(self.mLogR, ig1+1)), ig1)

        # For each grid point between ig1+1 and ig+2, compute integral from R[ig] to min(lnx2, R[ig+1]) at grid point ig
        # for ig in range(ig1+1, ig2+1):
        #     I += integral(self.mLogR[ig], min(lnx2, self.mLogR[ig+1]), ig)

        def body_fun(I, i):
            return cond((i >= ig1+1) & (i < ig2+1),
                        lambda _: I + integral(jnp.take(self.mLogR, i), jnp.minimum(lnx2, jnp.take(self.mLogR, i+1)), i),
                        lambda _: I,
                        None), None

        I, _ = scan(body_fun, I, jnp.arange(self.mLogR.shape[0]))
        
        # If lnx2 > R[ig2+1], compute integral from R[ig2+1] to lnx2 at grid point ig2
        # if lnx2 > self.mLogR[ig2+1]:
        #     I += integral(self.mLogR[ig2+1], lnx2, ig2)

        def true_fun(_):
            return I + integral(jnp.take(self.mLogR, ig2+1), lnx2, ig2)

        def false_fun(_):
            return I

        I = cond(lnx2 > jnp.take(self.mLogR, ig2+1), true_fun, false_fun, None)

        return I

    # find grid point ig such that R[ig] <= x <= R[ig+1]
    def find_grid_point(self, value): # Checked: this works correctly. 
        result = jnp.searchsorted(self.mLogR, value, side="left")-1
        
        # This is the issue! some sort of tracer when we needa concerte value. unsure how to fix
        # if result >= len(self.mLogR)-1:
        #     result = len(self.mLogR)-2

        def true_fun(_):
            return self.mLogR.shape[0]-2

        def false_fun(_):
            return result

        result = cond(result >= self.mLogR.shape[0]-1, true_fun, false_fun, None)

        return result