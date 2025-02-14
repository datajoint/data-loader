import pathlib
from collections import OrderedDict
from datetime import datetime

import numpy as np

_suite2p_ftypes = (
    "ops",
    "Fneu",
    "Fneu_chan2",
    "F",
    "F_chan2",
    "iscell",
    "spks",
    "stat",
    "redcell",
)


class Suite2p:
    """Wrapper class containing all suite2p outputs from one suite2p analysis routine.

    This wrapper includes outputs from the individual plane, with plane indexing
    starting from 0 Plane index of -1 indicates a suite2p "combined" outputs from all
    planes, thus saved in the "planes_combined" attribute. See also PlaneSuite2p class.

    Directory example:
        - plane0: ops.npy, F.npy, etc.
        - plane1: ops.npy, F.npy, etc.
        - combined: ops.npy, F.npy, etc.

    Example:
        > loaded_dataset = suite2p_loader.Suite2p(output_dir)

    """

    def __init__(self, suite2p_dir: str):
        """Initialize Suite2p class

        Args:
            suite2p_dir (str): Suite2p directory

        Raises:
            FileNotFoundError: Could not find Suite2p results
        """
        self.suite2p_dir = pathlib.Path(suite2p_dir)

        ops_filepaths = list(self.suite2p_dir.rglob("*ops.npy"))

        if not len(ops_filepaths):
            raise FileNotFoundError(
                "Suite2p output result files not found at {}".format(suite2p_dir)
            )

        self.planes = {}
        self.planes_combined = None
        for ops_fp in ops_filepaths:
            plane_s2p = PlaneSuite2p(ops_fp.parent)
            if plane_s2p.plane_idx == -1:
                self.planes_combined = plane_s2p
            else:
                self.planes[plane_s2p.plane_idx] = plane_s2p
        self.planes = OrderedDict({k: self.planes[k] for k in sorted(self.planes)})

        self.creation_time = min(
            [p.creation_time for p in self.planes.values()]
        )  # earliest file creation time
        self.curation_time = max(
            [p.curation_time for p in self.planes.values()]
        )  # most recent curation time


class PlaneSuite2p:
    """Parse the suite2p output directory and load data, ***per plane***.

    Suite2p output doc: https://suite2p.readthedocs.io/en/latest/outputs.html

    Expecting the following files:
        - ops:        Options file
        - Fneu:       Neuropil traces file for functional channel
        - Fneu_chan2: Neuropil traces file for channel 2
        - F:          Fluorescence traces for functional channel
        - F_chan2:    Fluorescence traces for channel 2
        - iscell:     Array of (user curated) cells and probability of being a cell
        - spks:       Spikes (raw deconvolved with OASIS package)
        - stat:       Various statistics for each cell
        - redcell:    "Red cell" (second channel) stats

    Attributes:
        alignment_channel: ops["align_by_chan"] as zero-indexed
        cell_prob:
        correlation_map: ops["Vcorr"]
        creation_time: earliest file creation time across planes
        curation_time: latest curation time across planes
        F: Fluorescence traces for functional channel as numpy array if exists
            If does not exist, returns empty list
        F_chan2: Fluorescence traces for channel 2 as numpy array if exists
            If does not exist, returns empty lists
        Fneu: Neuropil traces file for functional channel as numpy array if exists
            If does not exist, returns empty list
        Fneu_chan2: Neuropil traces file for channel 2 as numpy array if exists
            If does not exist, returns empty list
        fpath: path to plane folder
        iscell:
        max_proj_image: ops["max_proj"] if exists. Else np.full_like(mean_image))
        mean_image: ops["meanImg"]
        ops: Options file as numpy array
        plane_idx: plane index. -1 if combined, else number in path
        redcell: "Red cell" (second channel) stats as numpy array if exists
            If does not exist, returns empty list
        ref_image: ops["refImg"]
        segmentation_channel: ops["functional_chan"] as zero-indexed
        spks: Spikes (raw deconvolved with OASIS package) as numpy array if exists
            If does not exist, returns empty lists
        stat:  Various statistics for each cell as numpy array if exists
            If does not exist, returns empty lists
    """

    def __init__(self, suite2p_plane_dir: str):
        """Initialize PlaneSuite2p class given a plane directory

        Args:
            suite2p_plane_dir (str): Suite2p plane directory

        Raises:
            FileNotFoundError: No "ops.npy" found. Invalid suite2p plane folder
            FileNotFoundError: No "iscell.npy" found. Invalid suite2p plane folder
        """
        self.fpath = pathlib.Path(suite2p_plane_dir)

        # -- Verify dataset exists --
        ops_fp = self.fpath / "ops.npy"
        if not ops_fp.exists():
            raise FileNotFoundError(
                'No "ops.npy" found. Invalid suite2p plane folder: {}'.format(
                    self.fpath
                )
            )
        self.creation_time = datetime.fromtimestamp(ops_fp.stat().st_ctime)

        # -- Initialize attributes --
        for s2p_type in _suite2p_ftypes:
            setattr(self, "_{}".format(s2p_type), None)
        self._cell_prob = None

        self.plane_idx = (
            -1
            if self.fpath.name == "combined"
            else int(self.fpath.name.replace("plane", ""))
        )

    # -- load core files --

    @property
    def curation_time(self):
        print(
            "DeprecationWarning: 'curation_time' is deprecated, set to be the same as 'creation time', no longer reliable."
        )
        return self.creation_time

    @property
    def ops(self):
        if self._ops is None:
            fp = self.fpath / "ops.npy"
            self._ops = np.load(fp, allow_pickle=True).item()
        return self._ops

    @property
    def Fneu(self):
        if self._Fneu is None:
            fp = self.fpath / "Fneu.npy"
            self._Fneu = np.load(fp) if fp.exists() else []
        return self._Fneu

    @property
    def Fneu_chan2(self):
        if self._Fneu_chan2 is None:
            fp = self.fpath / "Fneu_chan2.npy"
            self._Fneu_chan2 = np.load(fp) if fp.exists() else []
        return self._Fneu_chan2

    @property
    def F(self):
        if self._F is None:
            fp = self.fpath / "F.npy"
            self._F = np.load(fp) if fp.exists() else []
        return self._F

    @property
    def F_chan2(self):
        if self._F_chan2 is None:
            fp = self.fpath / "F_chan2.npy"
            self._F_chan2 = np.load(fp) if fp.exists() else []
        return self._F_chan2

    @property
    def iscell(self):
        if self._iscell is None:
            fp = self.fpath / "iscell.npy"
            d = np.load(fp)
            self._iscell = d[:, 0].astype(bool)
            self._cell_prob = d[:, 1]
        return self._iscell

    @property
    def cell_prob(self):
        if self._cell_prob is None:
            fp = self.fpath / "iscell.npy"
            if fp.exists():
                d = np.load(fp)
                self._iscell = d[:, 0].astype(bool)
                self._cell_prob = d[:, 1]
        return self._cell_prob

    @property
    def spks(self):
        if self._spks is None:
            fp = self.fpath / "spks.npy"
            self._spks = np.load(fp) if fp.exists() else []
        return self._spks

    @property
    def stat(self):
        if self._stat is None:
            fp = self.fpath / "stat.npy"
            self._stat = np.load(fp, allow_pickle=True) if fp.exists() else []
        return self._stat

    @property
    def redcell(self):
        if self._redcell is None:
            fp = self.fpath / "redcell.npy"
            self._redcell = np.load(fp) if fp.exists() else []
        return self._redcell

    # -- image property --

    @property
    def ref_image(self):
        return self.ops["refImg"]

    @property
    def mean_image(self):
        return self.ops["meanImg"]

    @property
    def max_proj_image(self):
        return self.ops.get("max_proj", np.full_like(self.mean_image, np.nan))

    @property
    def correlation_map(self):
        return self.ops.get("VCorr", np.full_like(self.mean_image, np.nan))

    @property
    def alignment_channel(self):
        return self.ops["align_by_chan"] - 1  # suite2p is 1-based, convert to 0-based

    @property
    def segmentation_channel(self):
        return self.ops["functional_chan"] - 1  # suite2p is 1-based, convert to 0-based
