import cv2
import os
import pathlib
import shutil
import numpy as np
import multiprocessing

try:
    cv2.setNumThreads(0)
except:  # noqa E722
    pass  # TODO: remove bare except

try:
    import torch

    cuda_is_available = torch.cuda.is_available()
except:
    cuda_is_available = False
    pass

import caiman as cm
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.cnmf import CNMF

from .caiman_loader import _save_mc


def run_caiman(
    file_paths: list,
    parameters: dict,
    sampling_rate: float,
    output_dir: str,
    is3D: bool,
    n_processes: int = None,
):
    """
    Runs the standard caiman analysis pipeline (CNMF.fit_file method).

    Args:
        file_paths (list): Image (full) paths
        parameters (dict): Caiman parameters
        sampling_rate (float): Image sampling rate (Hz)
        output_dir (str): Output directory
        is3D (bool):  the data is 3D
    """
    parameters["is3D"] = is3D
    parameters["fnames"] = file_paths
    parameters["fr"] = sampling_rate

    use_cuda = parameters.get("use_cuda")
    parameters["use_cuda"] = cuda_is_available if use_cuda is None else use_cuda

    if "indices" in parameters:
        indices = parameters.pop(
            "indices"
        )  # Indices that restrict FOV for motion correction.
        indices = slice(*indices[0]), slice(*indices[1])
        parameters["motion"] = {**parameters.get("motion", {}), "indices": indices}

    caiman_temp = os.environ.get("CAIMAN_TEMP")
    os.environ["CAIMAN_TEMP"] = str(output_dir)

    # use 80% of available cores
    if n_processes is None:
        n_processes = int(np.floor(multiprocessing.cpu_count() * 0.8))
    _, dview, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=n_processes
    )

    try:
        opts = params.CNMFParams(params_dict=parameters)
        cnm = CNMF(n_processes, params=opts, dview=dview)
        cnmf_output, mc_output = cnm.fit_file(
            motion_correct=True,
            indices=None,  # Indices defined here restrict FOV for segmentation. `None` uses the full image for segmentation.
            include_eval=True,
            output_dir=output_dir,
            return_mc=True,
        )
    except Exception as e:
        dview.terminate()
        raise e
    else:
        cm.stop_server(dview=dview)

    if caiman_temp is not None:
        os.environ["CAIMAN_TEMP"] = caiman_temp
    else:
        del os.environ["CAIMAN_TEMP"]

    cnmf_output_file = pathlib.Path(cnmf_output.mmap_file[:-4] + "hdf5")
    cnmf_output_file = pathlib.Path(output_dir) / cnmf_output_file.name
    assert cnmf_output_file.exists()

    _save_mc(mc_output, cnmf_output_file.as_posix(), parameters["is3D"])
