from jaxtyping import Float, Int
from torch import Tensor
import torch
import numpy as np
from typing import TypedDict
import math
from pathlib import Path
from typing import Optional, Tuple, Union
from torchvision.io import decode_jpeg, read_file
from .corr_init import aggregate_confidences_and_warps, extract_keypoints_and_colors, select_best_keypoints, triangulate_points
from tqdm import tqdm

class Frame(TypedDict):
    K: Float[Tensor, "3 3"] # camera intrinsic matrix
    C2W: Float[Tensor, "4 4"] # camera to world transform matrix
    img: Float[Tensor, "3 H W"] | Path # the image or the path to the image

class GaussParamsDict(TypedDict):
    means: Float[torch.Tensor, "N 3"]
    opacities: Float[torch.Tensor, "N"]
    scales: Float[torch.Tensor, "N 3"]
    quats: Float[torch.Tensor, "N 4"]
    features_dc: Float[torch.Tensor, "N 3"]
    features_rest: Float[torch.Tensor, "N sh_bases 3"] # (SH + 1) ** 2 - 1

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def get_triangulated_points_as_gaussians(new_xyz: Float[Tensor, "N 3"], colors: Int[np.ndarray, "N 3"], camera_center: Float[Tensor, "3"], sh_degree: int, scaling_factor: float = 0.001, device = "cuda") -> GaussParamsDict:
    feature_rest_dim = (3 * ((sh_degree + 1)**2 - 1))
    N = len(new_xyz)
    dist_points_to_cam1 = torch.linalg.norm(camera_center.clone().detach().to(device) - new_xyz.to(device),
                                                dim=1, ord=2)
    gauss_params = GaussParamsDict(
        means=new_xyz.to(device),
        features_dc = RGB2SH(torch.tensor(colors.astype(np.float32) / 255.)).to(device),
        features_rest = torch.zeros((N, feature_rest_dim, 3)).to(device),
        opacities = torch.zeros(N).to(device),
        scales = torch.log((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 3)).to(device),
        quats = random_quat_tensor(N).to(device)
    )

    return gauss_params


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(fx, fy, width, height):
    fovX = focal2fov(fx, width)
    fovY = focal2fov(fy, height)
    zfar = 100.0
    znear = 0.01
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def load_image(image_path: Union[Path, str]) -> Float[Tensor, "3 W H"]:
    data = read_file(str(image_path))
    img_tensor = decode_jpeg(data)
    return img_tensor  # type: ignore

def get_roma_triangulated_points(
        source_frame: Frame,
        target_frames: list[Frame],
        roma_model,
        expansion_factor: int = 1,
        matches_per_reference: int = 20_000,
        keypoint_fit_error_tolerance: float = 0.01
    ):
    
    def get_full_proj_transform(frame: Frame, img_w: int, img_h: int):
        camera_params = {
            "fx": frame["K"][0, 0],
            "fy": frame["K"][1, 1],
            "img_w": img_w,
            "img_h": img_h
        }
        projection_matrix = getProjectionMatrix(camera_params["fx"], camera_params["fy"], camera_params["img_w"], camera_params["img_h"]).T
        frame_c2w = np.array(frame["C2W"])
        frame_w2c = np.float32(np.linalg.inv(frame_c2w)).T
        frame_w2c = torch.from_numpy(frame_w2c).float()
        return frame_w2c @ projection_matrix
    
    imgA = source_frame["img"] if isinstance(source_frame["img"], torch.Tensor) else load_image(source_frame["img"])
    closest_images = [frame["img"] if isinstance(frame["img"], torch.Tensor) else load_image(frame["img"]) for frame in target_frames]
    
    full_proj_transform = get_full_proj_transform(source_frame, imgA.shape[1], imgA.shape[2])
    full_proj_transforms_closest = [get_full_proj_transform(frame, img.shape[1], img.shape[2]) for frame, img in zip(target_frames, closest_images)]

    with torch.no_grad():
        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = aggregate_confidences_and_warps(
                imgA=imgA,
                closest_images=closest_images,
                roma_model=roma_model,
                verbose=False, output_dict={}
            )
    del closest_images

    # 2. Triangulate keypoints

    # Triangulate keypoints
    with torch.no_grad():
        matches = warps_max
        certainty = certainties_max
        certainty = certainty.clone()
        certainty[certainty > roma_model.sample_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )

        # Select based on certainty elements with high confidence. These are basically all of
        # kptsA_np.
        good_samples = torch.multinomial(certainty,
                                            num_samples=min(expansion_factor * matches_per_reference, len(certainty)),
                                            replacement=False)

        #certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all
    reference_image_dict = {
        "ref_image": imA,
        "NNs_images": imB_compound,
        "certainties_all": certainties_all,
        "warps_all": warps_all,
        "triangulated_points": [],
        "triangulated_points_errors_proj1": [],
        "triangulated_points_errors_proj2": []

    }
    with torch.no_grad():
        for NN_idx in tqdm(range(len(warps_all))):
            matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

            # Extract keypoints and colors
            kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
            )
            
            proj_matrices_A = full_proj_transform
            proj_matrices_B = full_proj_transforms_closest[NN_idx]

            triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                P1=torch.stack([proj_matrices_A] * matches_per_reference, dim=0),
                P2=torch.stack([proj_matrices_B] * matches_per_reference, dim=0),
                k1_x=kptsA_np[:matches_per_reference, 0], k1_y=kptsA_np[:matches_per_reference, 1],
                k2_x=kptsB_np[:matches_per_reference, 0], k2_y=kptsB_np[:matches_per_reference, 1])

            reference_image_dict["triangulated_points"].append(triangulated_points)
            reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
            reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)

    with torch.no_grad():
        NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
            NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
            NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
            NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
        
    good_points_mask = NNs_triangulated_points_selected_proj_errors <= keypoint_fit_error_tolerance
    new_xyz = NNs_triangulated_points_selected[good_points_mask, :-1]
    filtered_colors = kptsA_color[good_points_mask]
        
    return new_xyz, filtered_colors