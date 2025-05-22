from jaxtyping import Float, Int
from torch import Tensor
import torch
import numpy as np
from typing import TypedDict
from pathlib import Path
from .corr_init import select_cameras_kmeans, k_closest_vectors
from .utils import GaussParamsDict, get_triangulated_points_as_gaussians, Frame, get_roma_triangulated_points
from romatch import roma_outdoor, roma_indoor
from tqdm import tqdm

def dense_splat_init(
    frames: list[Frame],
    sh_degree: int = 3,

    matches_per_reference: int = 20000, # number of matches per reference(image)
    nns_per_ref: int = 3,   # number of nearest neighbors to consider for matching
    num_refs: int = 190, # number of references(images)
    scaling_factor: float = 0.001,
    device: str = "cuda",
    roma_model: str = "roma_outdoor",
    verbose: bool = False,
) -> GaussParamsDict:
    """
    Dense splat initialization for 3D Gaussian splatting.

    Args:
        frames: List of frames containing camera intrinsic matrix and image.
        matches_per_reference: Number of matches per reference(image).
        nns_per_ref: Number of nearest neighbors to consider for matching.
        num_refs: Number of references(images).
        scaling_factor: Scaling factor for the splatting.
        device: Device to run the model on.
        roma_model: Model to use for matching.
        verbose: Whether to print verbose output.
    Returns:
        GaussParamsDict: Dictionary containing the parameters of the Gaussians.
    """

    if roma_model == "roma_outdoor":
        roma_model = roma_outdoor(device=device)
    else:
        roma_model = roma_indoor(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False
    upper_threshold = roma_model.sample_thresh
    expansion_factor = 1
    keypoint_fit_error_tolerance = 0.01

    NUM_REFERENCE_FRAMES = min(len(frames), num_refs)
    NUM_NNS_PER_REF = min(len(frames), nns_per_ref)

    camera2world_matrices_flat = torch.stack([ torch.tensor(torch.linalg.inv(frame["C2W"])).transpose(0, 1).flatten() for frame in frames])
    
    selected_indices = select_cameras_kmeans(camera2world_matrices_flat, NUM_REFERENCE_FRAMES)
    print(len(selected_indices))
    closest_indices = k_closest_vectors(camera2world_matrices_flat, NUM_NNS_PER_REF)
    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

    for source_idx in tqdm(sorted(selected_indices)):
                # if source_idx > 10:
        #     break
        frame: Frame = frames[source_idx]

        frame_c2w = np.array(frame["C2W"])
        frame_w2c = np.linalg.inv(frame_c2w).T
        frame_w2c = torch.from_numpy(frame_w2c).float()
        camera_center = frame_w2c.inverse()[3, :3]

        print(source_idx)

        closest_frames = [frames[i] for i in closest_indices_selected[source_idx]]

        new_xyz, filtered_colors = get_roma_triangulated_points(
            frame,
            closest_frames,
            roma_model,
            expansion_factor,
            matches_per_reference,
            keypoint_fit_error_tolerance
        )    
        # 4. Save as gaussians
        with torch.no_grad():   
            gauss_params = get_triangulated_points_as_gaussians(new_xyz, filtered_colors, camera_center, sh_degree, scaling_factor)

            all_new_xyz.append(gauss_params["means"])
            all_new_features_dc.append(gauss_params["features_dc"])
            all_new_features_rest.append(gauss_params["features_rest"])
            all_new_opacities.append(gauss_params["opacities"])
            all_new_scaling.append(gauss_params["scales"])
            all_new_rotation.append(gauss_params["quats"])

    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    all_new_features_rest = torch.cat(all_new_features_rest, dim=0)
    all_new_opacities = torch.cat(all_new_opacities, dim=0)
    all_new_scaling = torch.cat(all_new_scaling, dim=0)
    all_new_rotation = torch.cat(all_new_rotation, dim=0)
    
    return GaussParamsDict(
        means=all_new_xyz,
        features_dc=all_new_features_dc,
        features_rest=all_new_features_rest,
        opacities=all_new_opacities,
        scales=all_new_scaling,
        quats=all_new_rotation
    )

    







    