import h5py
import numpy as np
import os


class Experiment:
    """
    Load example data provided e.g. at [1]_.

    The .h5 example data can be loaded directly
    using :code:`expt = Experiment('/path/to/file.h5')`.

    Parameters
    ----------
    path : str
        Path to the example data .h5 file

    Attributes
    ----------
    T_all : ndarray of shape (T, N)
        Matrix of extracted neural timeseries
    Y : ndarray
        Maximum intensity projection of volume
    centers : ndarray of shape (3, N)
        Centers of neurons in x, y, z order
    t : ndarray of shape (T, )
        Timestamps
    fhz : float
        Volume rate
    motion : ndarray of shape (T, npc)
        Matrix of the first npc motion PC timeseries, aligned to the neural data
    velocity_events : ndarray of shape (T, )
        Number of treadmill velocity events in each time bin
    out : str
        Path where analysis results are saved, :code:`os.path.dirname(path)`

    References
    ----------
    .. [1] Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F.,
           & Vaziri, A. (2024). Light beads microscopy recordings of up to one million neurons
           across mouse dorsal cortex during spontaneous behavior [Data set]. In Neuron. Zenodo.
           https://doi.org/10.5281/zenodo.10403684.
    """

    def __init__(self, path):
        assert os.path.basename(path)[-3:] == ".h5"
        print("Loading example data", os.path.basename(path))
        self._load_example_data(path)
        self.out = os.path.dirname(path)
        return

    def _load_example_data(self, path):
        with h5py.File(path, "r") as f:
            self.T_all = f["T_all"][()]
            (self.T, self.N) = self.T_all.shape
            self.nx = f["nx"][()]
            self.ny = f["ny"][()]
            self.nz = f["ny"][()]
            self.centers = np.squeeze(np.asarray([self.nx, self.ny, self.nz]))
            self.t = f["t"][()]
            self.duration = self.t[-1] - self.t[0]
            self.velocity_events = f["velocity_events"][()]
            self.motion = f["motion"][()]
            self.Y = f["Ym"][()]
            self.fhz = f["fhz"][()]
