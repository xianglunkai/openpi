import numpy as np  
from scipy.interpolate import make_lsq_spline, BSpline  


class BSplineFitter:
    """
    Performs Least-Squares B-spline fitting for fixed-length multi-dimensional time series.
    
    Parameters
    ----------
    T : int
        Number of timesteps (sequence length). Subsequent calls to `fit()` must use this length.
    k : int, default 3
        B-spline degree (degree = k).
    n_ctrl : int, default 8
        Number of control points. Must satisfy n_ctrl >= k + 1.
    """
    def __init__(self, T: int, k: int = 3, n_ctrl: int = 8):
        if n_ctrl < k + 1:
            raise ValueError("n_ctrl must be at least k + 1")

        self.T = T
        self.k = k
        self.n_ctrl = n_ctrl

        self.x = np.arange(T, dtype=float)

        # ----------- Knot vector t (open uniform / clamped) -----------
        # Number of internal knots = n_ctrl - k - 1
        t_internal = np.linspace(self.x[0], self.x[-1], n_ctrl - k + 1)
        # Repeat endpoints k times to ensure the B-spline is clamped (multiplicity k+1)
        self.t = np.concatenate(([self.x[0]] * k, t_internal,
                                 [self.x[-1]] * k)).astype(float)

        # ----------- Greville Abscissae -----------
        self.ctrl_x = np.array([self.t[j + 1: j + k + 1].mean()
                                for j in range(n_ctrl)],
                               dtype=np.float32)
        
        self._last_spline: BSpline | None = None

    def fit(self, y: np.ndarray):
        """
        Parameters
        ----------
        y : ndarray, shape (T, D)
            Data to be fitted. T must match the initialized length.
        
        Returns
        -------
        ctrl_y : ndarray, shape (n_ctrl, D), dtype float32
            Control point coefficients in the least-squares sense.
        """
        y = np.asarray(y, dtype=float)
        if y.shape[0] != self.T:
            raise ValueError(f"y must have length {self.T} on axis 0")

        spline = make_lsq_spline(self.x, y, self.t, self.k, axis=0)

        self._last_spline = spline

        ctrl_y = spline.c.astype(np.float32)    # shape (n_ctrl, D)
        return ctrl_y

    def rebuild(self,
                ctrl_y: np.ndarray | None = None,
                dtype=np.float32):
        """
        Reconstruct the full curve based on control point coefficients.

        If `ctrl_y` is None, use the spline from the most recent `fit()` call.

        Parameters
        ----------
        ctrl_y : ndarray or None
            Shape must be (n_ctrl,) or (n_ctrl, D). If None, 
            reconstructs using `_last_spline`.
        dtype : numpy dtype, default np.float32
            Data type of the returned values.

        Returns
        -------
        y_hat : ndarray, shape (T,) or (T, D)
            Estimated values at integer points 0 ... T-1.
        spline : BSpline
            The BSpline object for further evaluation or differentiation.
        """
        if ctrl_y is None:
            if self._last_spline is None:
                raise RuntimeError("No previous fit() result and ctrl_y is None.")
            spline = self._last_spline
        else:
            ctrl_y = np.asarray(ctrl_y)
            if ctrl_y.shape[0] != self.n_ctrl:
                raise ValueError(f"ctrl_y must have length {self.n_ctrl} on axis 0")
            spline = BSpline(self.t, ctrl_y, self.k, extrapolate=False)

        y_hat = spline(self.x).astype(dtype)
        return y_hat, spline

    
    def fit_batch(self, y_batch: np.ndarray):
        """
        y_batch : ndarray, shape (B, T, D)    
        return
        -------
        ctrl_y_batch : ndarray, shape (B, n_ctrl, D), dtype float32
        """
        y_batch = np.asarray(y_batch, dtype=float)
        if y_batch.ndim == 2:          # (B, T) → (B, T, 1)
            y_batch = y_batch[..., None]

        if y_batch.shape[1] != self.T:
            raise ValueError(f"time dimension must be {self.T} (axis 1)")

        B, _, D = y_batch.shape

        y_perm = np.transpose(y_batch, (1, 0, 2))       # (T, B, D)
        y_2d = y_perm.reshape(self.T, B * D)            # (T, B*D)

        spline = make_lsq_spline(self.x, y_2d, self.t, self.k, axis=0)

        c = spline.c.reshape(self.n_ctrl, B, D)          # (n_ctrl, B, D)
        ctrl_y_batch = np.transpose(c, (1, 0, 2))        # (B, n_ctrl, D)

        ctrl_y_batch = np.ascontiguousarray(ctrl_y_batch, dtype=np.float32)
        self._last_spline = spline

        return ctrl_y_batch