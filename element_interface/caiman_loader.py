import os
import pathlib
from datetime import datetime
import re
import caiman as cm
import h5py
import numpy as np
import scipy
from tqdm import tqdm

_required_hdf5_fields = [
    "/motion_correction/reference_image",
    "/motion_correction/correlation_image",
    "/motion_correction/average_image",
    "/motion_correction/max_image",
    "/estimates/A",
]


class CaImAn:
    """
    Loader class for CaImAn analysis results
    A top level aggregator of multiple set of CaImAn results (e.g. multi-plane analysis)
    Calling _CaImAn (see below) under the hood
    """

    def __init__(self, caiman_dir: str):
        """Initialize CaImAn loader class

        Args:
            caiman_dir (str): string, absolute file path to CaIman directory

        Raises:
            FileNotFoundError: No CaImAn analysis output file found
            FileNotFoundError: No CaImAn analysis output found, missing required fields
        """
        # ---- Search and verify CaImAn output file exists ----
        caiman_dir = pathlib.Path(caiman_dir)
        if not caiman_dir.exists():
            raise FileNotFoundError("CaImAn directory not found: {}".format(caiman_dir))

        caiman_subdirs = []
        for fp in caiman_dir.rglob("*.hdf5"):
            with h5py.File(fp, "r") as h5f:
                if all(s in h5f for s in _required_hdf5_fields):
                    caiman_subdirs.append(fp.parent)

        if not caiman_subdirs:
            raise FileNotFoundError(
                "No CaImAn analysis output file found at {}"
                " containg all required fields ({})".format(
                    caiman_dir, _required_hdf5_fields
                )
            )

        # Extract CaImAn results from all planes, sorted by plane index
        _planes_caiman = {}
        for idx, caiman_subdir in enumerate(sorted(caiman_subdirs)):
            pln_cm = _CaImAn(caiman_subdir.as_posix())
            pln_idx_match = re.search(r"pln(\d+)_.*", caiman_subdir.stem)
            pln_idx = pln_idx_match.groups()[0] if pln_idx_match else idx
            pln_cm.plane_idx = pln_idx
            _planes_caiman[pln_idx] = pln_cm
        sorted_pln_ind = sorted(list(_planes_caiman.keys()))
        self.planes = {k: _planes_caiman[k] for k in sorted_pln_ind}

        self.creation_time = min(
            [p.creation_time for p in self.planes.values()]
        )  # ealiest file creation time
        self.curation_time = max(
            [p.curation_time for p in self.planes.values()]
        )  # most recent curation time

        # is this 3D CaImAn analyis or multiple 2D per-plane analysis
        if len(self.planes) > 1:
            # if more than one set of caiman result, likely to be multiple 2D per-plane
            # assert that the "is3D" value are all False for each of the caiman result
            assert all(p.params.motion["is3D"] is False for p in self.planes.values())
            self.is3D = False
            self.is_multiplane = True
        else:
            self.is3D = list(self.planes.values())[0].params.motion["is3D"]
            self.is_multiplane = False

        if self.is_multiplane and self.is3D:
            raise NotImplementedError(
                f"Unable to load CaImAn results mixed between 3D and multi-plane analysis"
            )

        self._motion_correction = None
        self._masks = None
        self._ref_image = None
        self._mean_image = None
        self._max_proj_image = None
        self._correlation_map = None

    @property
    def is_pw_rigid(self):
        pw_rigid = set(p.params.motion["pw_rigid"] for p in self.planes.values())
        assert (
            len(pw_rigid) == 1
        ), f"Unable to load CaImAn results mixed between rigid and pw_rigid motion correction"
        return pw_rigid.pop()

    @property
    def motion_correction(self):
        if self._motion_correction is None:
            self._motion_correction = (
                self.extract_pw_rigid_mc()
                if self.is_pw_rigid
                else self.extract_rigid_mc()
            )
        return self._motion_correction

    def extract_rigid_mc(self):
        # -- rigid motion correction --
        rigid_correction = {}
        for pln_idx, (plane, pln_cm) in enumerate(self.planes.items()):
            if pln_idx == 0:
                rigid_correction = {
                    "x_shifts": pln_cm.motion_correction["shifts_rig"][:, 0],
                    "y_shifts": pln_cm.motion_correction["shifts_rig"][:, 1],
                }
                rigid_correction["x_std"] = np.nanstd(
                    rigid_correction["x_shifts"].flatten()
                )
                rigid_correction["y_std"] = np.nanstd(
                    rigid_correction["y_shifts"].flatten()
                )
            else:
                rigid_correction["x_shifts"] = np.vstack(
                    [
                        rigid_correction["x_shifts"],
                        pln_cm.motion_correction["shifts_rig"][:, 0],
                    ]
                )
                rigid_correction["x_std"] = np.nanstd(
                    rigid_correction["x_shifts"].flatten()
                )
                rigid_correction["y_shifts"] = np.vstack(
                    [
                        rigid_correction["y_shifts"],
                        pln_cm.motion_correction["shifts_rig"][:, 1],
                    ]
                )
                rigid_correction["y_std"] = np.nanstd(
                    rigid_correction["y_shifts"].flatten()
                )

        if not self.is_multiplane:
            pln_cm = list(self.planes.values())[0]
            rigid_correction["z_shifts"] = (
                pln_cm.motion_correction["shifts_rig"][:, 2]
                if self.is3D
                else np.full_like(rigid_correction["x_shifts"], 0)
            )
            rigid_correction["z_std"] = (
                np.nanstd(pln_cm.motion_correction["shifts_rig"][:, 2])
                if self.is3D
                else np.nan
            )
        else:
            rigid_correction["z_shifts"] = np.full_like(rigid_correction["x_shifts"], 0)
            rigid_correction["z_std"] = np.nan

        rigid_correction["outlier_frames"] = None

        return rigid_correction

    def extract_pw_rigid_mc(self):
        # -- piece-wise rigid motion correction --
        nonrigid_correction, nonrigid_blocks = {}
        for pln_idx, (plane, pln_cm) in enumerate(self.planes.items()):
            block_count = len(nonrigid_blocks)
            if pln_idx == 0:
                nonrigid_correction = {
                    "block_height": (
                        pln_cm.params.motion["strides"][0]
                        + pln_cm.params.motion["overlaps"][0]
                    ),
                    "block_width": (
                        pln_cm.params.motion["strides"][1]
                        + pln_cm.params.motion["overlaps"][1]
                    ),
                    "block_depth": 1,
                    "block_count_x": len(
                        set(pln_cm.motion_correction["coord_shifts_els"][:, 0])
                    ),
                    "block_count_y": len(
                        set(pln_cm.motion_correction["coord_shifts_els"][:, 2])
                    ),
                    "block_count_z": len(self.planes),
                    "outlier_frames": None,
                }
            for b_id in range(len(pln_cm.motion_correction["x_shifts_els"][0, :])):
                b_id += block_count
                nonrigid_blocks[b_id] = {
                    "block_id": b_id,
                    "block_x": np.arange(
                        *pln_cm.motion_correction["coord_shifts_els"][b_id, 0:2]
                    ),
                    "block_y": np.arange(
                        *pln_cm.motion_correction["coord_shifts_els"][b_id, 2:4]
                    ),
                    "block_z": (
                        np.arange(
                            *pln_cm.motion_correction["coord_shifts_els"][b_id, 4:6]
                        )
                        if self.is3D
                        else np.full_like(
                            np.arange(
                                *pln_cm.motion_correction["coord_shifts_els"][b_id, 0:2]
                            ),
                            pln_idx,
                        )
                    ),
                    "x_shifts": pln_cm.motion_correction["x_shifts_els"][:, b_id],
                    "y_shifts": pln_cm.motion_correction["y_shifts_els"][:, b_id],
                    "z_shifts": (
                        pln_cm.motion_correction["z_shifts_els"][:, b_id]
                        if self.is3D
                        else np.full_like(
                            pln_cm.motion_correction["x_shifts_els"][:, b_id],
                            0,
                        )
                    ),
                    "x_std": np.nanstd(
                        pln_cm.motion_correction["x_shifts_els"][:, b_id]
                    ),
                    "y_std": np.nanstd(
                        pln_cm.motion_correction["y_shifts_els"][:, b_id]
                    ),
                    "z_std": (
                        np.nanstd(pln_cm.motion_correction["z_shifts_els"][:, b_id])
                        if self.is3D
                        else np.nan
                    ),
                }

        if not self.is_multiplane and self.is3D:
            pln_cm = list(self.planes.values())[0]
            nonrigid_correction["block_depth"] = (
                pln_cm.params.motion["strides"][2] + pln_cm.params.motion["overlaps"][2]
            )
            nonrigid_correction["block_count_z"] = len(
                set(pln_cm.motion_correction["coord_shifts_els"][:, 4])
            )

        return nonrigid_correction, nonrigid_blocks

    @property
    def masks(self):
        if self._masks is None:
            all_masks = []
            for pln_idx, pln_cm in sorted(self.planes.items()):
                mask_count = len(all_masks)  # increment mask id from all "plane"
                all_masks.extend(
                    [
                        {
                            **m,
                            "mask_id": m["mask_id"] + mask_count,
                            "orig_mask_id": m["mask_id"],
                            "accepted": (
                                m["mask_id"] in pln_cm.cnmf.estimates.idx_components
                                if pln_cm.cnmf.estimates.idx_components is not None
                                else False
                            ),
                        }
                        for m in pln_cm.masks
                    ]
                )

            self._masks = all_masks
        return self._masks

    @property
    def alignment_channel(self):
        return 0  # hard-code to channel index 0

    @property
    def segmentation_channel(self):
        return 0  # hard-code to channel index 0

    # -- image property --

    def _get_image(self, img_type):
        if not self.is_multiplane:
            pln_cm = list(self.planes.values())[0]
            img_ = (
                pln_cm.motion_correction[img_type].transpose()
                if self.is3D
                else pln_cm.motion_correction[img_type][...][..., np.newaxis]
            )
        else:
            img_ = np.dstack(
                [
                    pln_cm.motion_correction[img_type][...]
                    for pln_cm in self.planes.values()
                ]
            )
        return img_

    @property
    def ref_image(self):
        if self._ref_image is None:
            self._ref_image = self._get_image("reference_image")
        return self._ref_image

    @property
    def mean_image(self):
        if self._mean_image is None:
            self._mean_image = self._get_image("average_image")
        return self._mean_image

    @property
    def max_proj_image(self):
        if self._max_proj_image is None:
            self._max_proj_image = self._get_image("max_image")
        return self._max_proj_image

    @property
    def correlation_map(self):
        if self._correlation_map is None:
            self._correlation_map = self._get_image("correlation_image")
        return self._correlation_map


class _CaImAn:
    """Parse the CaImAn output file

    [CaImAn results doc](https://caiman.readthedocs.io/en/master/Getting_Started.html#result-variables-for-2p-batch-analysis)

    Expecting the following objects:
        - dims:
        - dview:
        - estimates:              Segmentations and traces
        - mmap_file:
        - params:                 Input parameters
        - remove_very_bad_comps:
        - skip_refinement:
        - motion_correction:      Motion correction shifts and summary images

    Example:
        > output_dir = '<imaging_root_data_dir>/subject1/session0/caiman'

        > loaded_dataset = caiman_loader.CaImAn(output_dir)

    Attributes:
        alignment_channel: hard-coded to 0
        caiman_fp: file path with all required files:
            "/motion_correction/reference_image",
            "/motion_correction/correlation_image",
            "/motion_correction/average_image",
            "/motion_correction/max_image",
            "/estimates/A",
        cnmf: loaded caiman object; cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_fp)
        creation_time: file creation time
        curation_time: file creation time
        extract_masks: function to extract masks
        h5f: caiman_fp read as h5py file
        masks: dict result of extract_masks
        motion_correction: h5f "motion_correction" property
        params: cnmf.params
        segmentation_channel: hard-coded to 0
        plane_idx: N/A if `is3D` else hard-coded to 0
    """

    def __init__(self, caiman_dir: str):
        """Initialize CaImAn loader class

        Args:
            caiman_dir (str): string, absolute file path to CaIman directory

        Raises:
            FileNotFoundError: No CaImAn analysis output file found
            FileNotFoundError: No CaImAn analysis output found, missing required fields
        """
        # ---- Search and verify CaImAn output file exists ----
        caiman_dir = pathlib.Path(caiman_dir)
        if not caiman_dir.exists():
            raise FileNotFoundError("CaImAn directory not found: {}".format(caiman_dir))

        for fp in caiman_dir.glob("*.hdf5"):
            with h5py.File(fp, "r") as h5f:
                if all(s in h5f for s in _required_hdf5_fields):
                    self.caiman_fp = fp
                    break
        else:
            raise FileNotFoundError(
                "No CaImAn analysis output file found at {}"
                " containing all required fields ({})".format(
                    caiman_dir, _required_hdf5_fields
                )
            )

        # ---- Initialize CaImAn's results ----
        self.cnmf = cm.source_extraction.cnmf.cnmf.load_CNMF(self.caiman_fp)
        self.params = self.cnmf.params

        self.h5f = h5py.File(self.caiman_fp, "r")
        self.plane_idx = None if self.params.motion["is3D"] else 0
        self._motion_correction = None
        self._masks = None

        # ---- Metainfo ----
        self.creation_time = datetime.fromtimestamp(os.stat(self.caiman_fp).st_ctime)
        self.curation_time = datetime.fromtimestamp(os.stat(self.caiman_fp).st_ctime)

    @property
    def motion_correction(self):
        if self._motion_correction is None:
            self._motion_correction = self.h5f["motion_correction"]
        return self._motion_correction

    @property
    def masks(self):
        if self._masks is None:
            self._masks = self.extract_masks()
        return self._masks

    @property
    def alignment_channel(self):
        return 0  # hard-code to channel index 0

    @property
    def segmentation_channel(self):
        return 0  # hard-code to channel index 0

    def extract_masks(self) -> dict:
        """Extract masks from CaImAn object

        Raises:
            NotImplemented: Not yet implemented for 3D datasets

        Returns:
            dict: Mask attributes - mask_id, mask_npix, mask_weights,
                mask_center_x, mask_center_y, mask_center_z,
                mask_xpix, mask_ypix, mask_zpix, inferred_trace, dff, spikes
        """
        if self.params.motion["is3D"]:
            raise NotImplementedError(
                "CaImAn mask extraction for volumetric data not yet implemented"
            )

        comp_contours = cm.utils.visualization.get_contours(
            self.cnmf.estimates.A, self.cnmf.dims
        )

        masks = []
        for comp_idx, comp_contour in enumerate(comp_contours):
            ind, _, weights = scipy.sparse.find(self.cnmf.estimates.A[:, comp_idx])
            if self.cnmf.params.motion["is3D"]:
                xpix, ypix, zpix = np.unravel_index(ind, self.cnmf.dims, order="F")
                center_x, center_y, center_z = comp_contour["CoM"].astype(int)
            else:
                xpix, ypix = np.unravel_index(ind, self.cnmf.dims, order="F")
                center_x, center_y = comp_contour["CoM"].astype(int)
                center_z = self.plane_idx
                zpix = np.full(len(weights), center_z)

            masks.append(
                {
                    "mask_id": comp_contour["neuron_id"],
                    "mask_npix": len(weights),
                    "mask_weights": weights,
                    "mask_center_x": center_x,
                    "mask_center_y": center_y,
                    "mask_center_z": center_z,
                    "mask_xpix": xpix,
                    "mask_ypix": ypix,
                    "mask_zpix": zpix,
                    "inferred_trace": self.cnmf.estimates.C[comp_idx, :],
                    "dff": self.cnmf.estimates.F_dff[comp_idx, :],
                    "spikes": self.cnmf.estimates.S[comp_idx, :],
                }
            )
        return masks


def _process_scanimage_tiff(scan_filenames, output_dir="./", split_depths=False):
    """
    Read ScanImage TIFF - reshape into volumetric data based on scanning depths/channels
    Save new TIFF files for each channel - with shape (frame x height x width x depth)
    """
    import scanreader
    from tifffile import imsave

    # ------------ CaImAn multi-channel multi-plane tiff file ------------
    for scan_filename in tqdm(scan_filenames):
        scan = scanreader.read_scan(scan_filename)
        cm_movie = cm.load(scan_filename)

        # ---- Volumetric movie: (depth x height x width x channel x frame) ----
        # tiff pages are ordered as:
        # ch0-pln0-t0, ch1-pln0-t0, ch0-pln1-t0, ch1-pln1-t0, ..., ch0-pln1-t5, ch1-pln1-t5, ...

        vol_timeseries = np.full(
            (
                scan.num_scanning_depths,
                scan.image_height,
                scan.image_width,
                scan.num_channels,
                scan.num_frames,
            ),
            0,
        ).astype(scan.dtype)
        for pln_idx in range(scan.num_scanning_depths):
            for chn_idx in range(scan.num_channels):
                pln_chn_ind = np.arange(
                    pln_idx * scan.num_channels + chn_idx,
                    scan._num_pages,
                    scan.num_scanning_depths * scan.num_channels,
                )
                vol_timeseries[pln_idx, :, :, chn_idx, :] = cm_movie[
                    pln_chn_ind, :, :
                ].transpose(1, 2, 0)

        # save volumetric movie for individual channel
        output_dir = pathlib.Path(output_dir)
        fname = pathlib.Path(scan_filename).stem

        for chn_idx in range(scan.num_channels):
            if scan.num_scanning_depths == 1:
                chn_vol = (
                    vol_timeseries[0, :, :, chn_idx, :].squeeze().transpose(2, 0, 1)
                )  # (frame x height x width)
            else:
                chn_vol = vol_timeseries[:, :, :, chn_idx, :].transpose(
                    3, 1, 2, 0
                )  # (frame x height x width x depth)
            save_fp = output_dir / "{}_chn{}.tif".format(fname, chn_idx)
            imsave(save_fp.as_posix(), chn_vol)


def _save_mc(
    mc,
    caiman_fp: str,
    is3D: bool,
    summary_images: dict = None,
):
    """Save motion correction to hdf5 output

    Run these commands after the CaImAn analysis has completed.
    This will save the relevant motion correction data into the '*.hdf5' file.
    Please do not clear variables from memory prior to running these commands.
    The motion correction (mc) object will be read from memory.

    Args:
        mc : CaImAn motion correction object, including the following properties:
            shifts_rig :   Rigid transformation x and y shifts per frame
            x_shifts_els : Non rigid transformation x shifts per frame per block
            y_shifts_els : Non rigid transformation y shifts per frame per block
        caiman_fp (str): CaImAn output (*.hdf5) file path - append if exists, else create new one
        is3D (bool):  the data is 3D
        summary_images(dict): dict of summary images (average_image, max_image, correlation_image) - if None, will be computed, if provided as empty dict, will not be computed
    """
    Yr, dims, T = cm.mmapping.load_memmap(mc.mmap_file[0])
    # Load the first frame of the movie
    mc_image = np.reshape(Yr[: np.product(dims), :1], [1] + list(dims), order="F")

    # Compute mc.coord_shifts_els
    grid = []
    if is3D:
        for _, _, _, x, y, z, _ in cm.motion_correction.sliding_window_3d(
            mc_image[0, :, :, :], mc.overlaps, mc.strides
        ):
            grid.append(
                [
                    x,
                    x + mc.overlaps[0] + mc.strides[0],
                    y,
                    y + mc.overlaps[1] + mc.strides[1],
                    z,
                    z + mc.overlaps[2] + mc.strides[2],
                ]
            )
    else:
        for _, _, x, y, _ in cm.motion_correction.sliding_window(
            mc_image[0, :, :], mc.overlaps, mc.strides
        ):
            grid.append(
                [
                    x,
                    x + mc.overlaps[0] + mc.strides[0],
                    y,
                    y + mc.overlaps[1] + mc.strides[1],
                ]
            )

    # Open hdf5 file and create 'motion_correction' group
    caiman_fp = pathlib.Path(caiman_fp)
    h5f = h5py.File(caiman_fp, "r+" if caiman_fp.exists() else "w")
    h5g = h5f.require_group("motion_correction")

    # Write motion correction shifts and motion corrected summary images to hdf5 file
    if mc.pw_rigid:
        h5g.require_dataset(
            "x_shifts_els",
            shape=np.shape(mc.x_shifts_els),
            data=mc.x_shifts_els,
            dtype=mc.x_shifts_els[0][0].dtype,
        )
        h5g.require_dataset(
            "y_shifts_els",
            shape=np.shape(mc.y_shifts_els),
            data=mc.y_shifts_els,
            dtype=mc.y_shifts_els[0][0].dtype,
        )
        if is3D:
            h5g.require_dataset(
                "z_shifts_els",
                shape=np.shape(mc.z_shifts_els),
                data=mc.z_shifts_els,
                dtype=mc.z_shifts_els[0][0].dtype,
            )

        h5g.require_dataset(
            "coord_shifts_els", shape=np.shape(grid), data=grid, dtype=type(grid[0][0])
        )

        # For CaImAn, reference image is still a 2D array even for the case of 3D
        # Assume that the same ref image is used for all the planes
        reference_image = (
            np.tile(mc.total_template_els, (1, 1, dims[-1]))
            if is3D
            else mc.total_template_els
        )
    else:
        h5g.require_dataset(
            "shifts_rig",
            shape=np.shape(mc.shifts_rig),
            data=mc.shifts_rig,
            dtype=mc.shifts_rig[0][0].dtype,
        )
        h5g.require_dataset(
            "coord_shifts_rig", shape=np.shape(grid), data=grid, dtype=type(grid[0][0])
        )
        reference_image = (
            np.tile(mc.total_template_rig, (1, 1, dims[-1]))
            if is3D
            else mc.total_template_rig
        )

    if summary_images is None:
        # Load motion corrected mmap image
        mc_image = cm.load(mc.mmap_file, is3D=is3D)

        # Compute motion corrected summary images
        average_image = np.mean(mc_image, axis=0)
        max_image = np.max(mc_image, axis=0)

        # Compute motion corrected correlation image
        correlation_image = cm.local_correlations(
            mc_image.transpose((1, 2, 3, 0) if is3D else (1, 2, 0))
        )
        correlation_image[np.isnan(correlation_image)] = 0

        summary_images = {
            "average_image": average_image,
            "max_image": max_image,
            "correlation_image": correlation_image,
        }

    for img_type, img in summary_images.items():
        h5g.require_dataset(
            img_type,
            shape=np.shape(img),
            data=img,
            dtype=img.dtype,
        )

    h5g.require_dataset(
        "reference_image",
        shape=np.shape(reference_image),
        data=reference_image,
        dtype=reference_image.dtype,
    )

    # Close hdf5 file
    h5f.close()
