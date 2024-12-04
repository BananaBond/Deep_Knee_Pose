
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from beartype import beartype
import pandas as pd
import sys
import diffdrr
import ast
import nibabel as nib

from .calibration import RigidTransform, perspective_projection

@beartype
class KneeFitDataset(torch.utils.data.Dataset):
    """
    Get X-ray projections and poses from specimens in the `DeepFluoro` dataset.

    Given a specimen ID and projection index, returns the projection and the camera matrix for DiffDRR.
    """

    def __init__(
        self,
        id_number: int,  # Specimen number (1-6)
        filename: Optional[Union[str, Path]] = None,  # Path to DeepFluoro h5 file
        preprocess: bool = True,  # Preprocess X-rays
    ):
        # Load the volume
        (
            self.projections,
            self.volume,
            self.spacing,
            self.lps2volume,
            self.intrinsic,
            self.extrinsic,
            self.focal_len,
            self.x0,
            self.y0,
        ) = load_kneefit_dataset(id_number, filename)
        self.preprocess = preprocess

        # Get the isocenter pose (AP viewing angle at volume isocenter)
        # probably have to change the viewing angle here to LAT viewing angle
        # isocenter_rot = torch.tensor([[torch.pi / 2, torch.pi / 2, -torch.pi / 2]])
        isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
        isocenter_xyz = torch.tensor(self.volume.shape) * self.spacing / 2
        isocenter_xyz = isocenter_xyz.unsqueeze(0)
        self.isocenter_pose = RigidTransform(
            isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
        )

        # Camera matrices and fiducials for the specimen
        # self.fiducials = get_3d_fiducials(self.specimen)
        self.fiducials = random_3d_fiducials(self.volume)

        # Miscellaneous transformation matrices for wrangling SE(3) poses
        self.flip_xz = RigidTransform(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(3),
        )
        self.translate = RigidTransform(
            torch.eye(3),
            torch.tensor([-self.focal_len / 2, 0.0, 0.0]),
        )
        self.flip_180 = RigidTransform(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            torch.zeros(3),
        )

    def __len__(self):
        return len(self.projections)

    def __iter__(self):
        return iter(self[idx] for idx in range(len(self)))

    def __getitem__(self, idx):
        """
        (1) Swap the x- and z-axes
        (2) Reverse the x-axis to make the matrix E(3) -> SE(3)
        (3) Move the camera to the origin
        (4) Rotate the detector plane by 180, if offset
        (5) Form the full SE(3) transformation matrix
        """
        projection = self.projections[f"{idx:03d}"]
        img = torch.from_numpy(projection["image/pixels"][:])
        world2volume = torch.from_numpy(projection["gt-poses/cam-to-pelvis-vol"][:])
        world2volume = RigidTransform(world2volume[:3, :3], world2volume[:3, 3])
        pose = convert_deepfluoro_to_diffdrr(self, world2volume)

        # Handle rotations in the imaging dataset
        if self._rot_180_for_up(idx):
            img = torch.rot90(img, k=2)
            pose = self.flip_180.compose(pose)

        # Optionally, preprocess the images
        img = img.unsqueeze(0).unsqueeze(0)
        if self.preprocess:
            img = preprocess(img)

        return img, pose

    def get_2d_fiducials(self, idx, pose):
        # Get the fiducials from the true camera pose
        _, true_pose = self.__getitem__(idx)
        extrinsic = (
            self.lps2volume.inverse()
            .compose(true_pose.inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        true_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )

        # Get the fiducials from the predicted camera pose
        extrinsic = (
            self.lps2volume.inverse()
            .compose(pose.cpu().inverse())
            .compose(self.translate)
            .compose(self.flip_xz)
        )
        pred_fiducials = perspective_projection(
            extrinsic, self.intrinsic, self.fiducials
        )

        if self.preprocess:
            true_fiducials -= 50
            pred_fiducials -= 50

        return true_fiducials, pred_fiducials

    def _rot_180_for_up(self, idx):
        return self.projections[f"{idx:03d}"]["rot-180-for-up"][()]

def convert_deepfluoro_to_diffdrr(specimen, pose: RigidTransform):
    """Transform the camera coordinate system used in DeepFluoro to the convention used by DiffDRR."""
    return (
        specimen.translate.compose(specimen.flip_xz)
        .compose(specimen.extrinsic.inverse())
        .compose(pose)
        .compose(specimen.lps2volume.inverse())
    )


def convert_diffdrr_to_deepfluoro(specimen, pose: RigidTransform):
    """Transform the camera coordinate system used in DiffDRR to the convention used by DeepFluoro."""
    return (
        specimen.lps2volume.inverse()
        .compose(pose.inverse())
        .compose(specimen.translate)
        .compose(specimen.flip_xz)
    )

from torch.nn.functional import pad

from .calibration import perspective_projection


class Evaluator:
    def __init__(self, specimen, idx):
        # Save matrices to device
        self.translate = specimen.translate
        self.flip_xz = specimen.flip_xz
        self.intrinsic = specimen.intrinsic
        self.intrinsic_inv = specimen.intrinsic.inverse()

        # Get gt fiducial locations
        self.specimen = specimen
        self.fiducials = specimen.fiducials
        gt_pose = specimen[idx][1]
        self.true_projected_fiducials = self.project(gt_pose)

    def project(self, pose):
        extrinsic = convert_diffdrr_to_deepfluoro(self.specimen, pose)
        x = perspective_projection(extrinsic, self.intrinsic, self.fiducials)
        x = -self.specimen.focal_len * torch.einsum(
            "ij, bnj -> bni",
            self.intrinsic_inv,
            pad(x, (0, 1), value=1),  # Convert to homogenous coordinates
        )
        extrinsic = (
            self.flip_xz.inverse().compose(self.translate.inverse()).compose(pose)
        )
        return extrinsic.transform_points(x)

    def __call__(self, pose):
        pred_projected_fiducials = self.project(pose)
        registration_error = (
            (self.true_projected_fiducials - pred_projected_fiducials)
            .norm(dim=-1)
            .mean()
        )
        registration_error *= 0.194  # Pixel spacing is 0.194 mm / pixel isotropic
        return registration_error

from diffdrr.utils import parse_intrinsic_matrix, get_principal_point


def load_kneefit_dataset(id_number, filename):
    # Open the H5 file for the dataset
    if filename is None:
        root = Path(__file__).parent.parent.absolute()
        filename = root / "data/sipla_bone_preprocessed.csv"
    f = pd.read_csv(filename)
    assert id_number in range(1, f["patient_id"].nunique())
    f = f[f["patient_id"] == id_number]
    (
        intrinsic,
        extrinsic,
        num_cols,
        num_rows,
        proj_col_spacing,
        proj_row_spacing,
    ) = parse_proj_params(f)

    # uncomment if focal length is in unit length
    focal_len = intrinsic[0,0]
    x0, y0 = intrinsic[0,2], intrinsic[1,2]

    # uncomment if focal length is in pixels
    # focal_len, x0, y0 = parse_intrinsic_matrix(
    #     intrinsic,
    #     num_rows,
    #     num_cols,
    #     proj_row_spacing,
    #     proj_col_spacing,
    # )


    # Try to load the particular specimen
    projections = f[f["patient_id"] == id_number]

    # Parse the volume
    volume, spacing, lps2volume = parse_volume(projections)
    return (
        projections,
        volume,
        spacing,
        lps2volume,
        intrinsic,
        extrinsic,
        focal_len,
        x0,
        y0,
    )

def get_volume_data(volume):
    nii_data = nib.load(volume)
    # Access the affine transformation matrix
    # affine = nii_data.affine
    # Access the image data as a NumPy array
    data = nii_data.get_fdata()
    # Access the affine transformation matrix
    affine = nii_data.affine
    return data, affine


def parse_volume(specimen):
    # Parse the volume
    #use pitch as spacing if using stl converted to voxel grid OR use code from nii library nibabel from data_analysis notebook
    # spacing = specimen["vol/spacing"][:].flatten()
    spacing = np.array(specimen["spacing"].unique()[0])
    #check shape of 3d array
    # import ipdb; ipdb.set_trace()
    volume, affine = get_volume_data(specimen["femur_nii"].unique()[0])

    volume = volume.astype(np.float32)
    # volume = specimen["vol/pixels"][:].astype(np.float32)

    #is swapping the axis same as changing lateral to AP view?
    volume = np.swapaxes(volume, 0, 2)[::-1].copy()

    # Parse the translation matrix from LPS coordinates to volume coordinates
    origin = torch.tensor(affine[:3, 3])
    lps2volume = RigidTransform(torch.eye(3), origin)
    return volume, spacing, lps2volume


def parse_proj_params(f):
    # proj_params = f["proj-params"]
    # extrinsic parameters here do not matter, set to zeros
    try:
        fx = f["cal_focal_length"].unique()[0]
        px = f["cal_principalp_x"].unique()[0]
        py = f["cal_principalp_y"].unique()[0]
    except Exception as err:
        print("camera instrinsics must be the same for all frames of 1 patient")
        sys.exit()
    extrinsic = torch.eye((4,4), dtype=torch.float32)
    extrinsic[(0, 1, 2), 3] = (fx, px, py)

    #understand why extrinsic goes through RigidTransform
    extrinsic = RigidTransform(extrinsic[..., :3, :3], extrinsic[:3, 3])
    intrinsic = torch.tensor([[fx, 0, px],
                              [0, fx, py],
                              [0, 0, 1]], dtype=torch.float32)
    num_cols = num_rows = 1000
    proj_col_spacing = proj_row_spacing = float(f["cal_mm_per_pxl"].unique()[0])
    return intrinsic, extrinsic, num_cols, num_rows, proj_col_spacing, proj_row_spacing

def is_within_surface(nii_array, point, surface_threshold):
    x, y, z = point
    # Example condition: Check if the voxel value is above the surface threshold
    if nii_array[x, y, z] > surface_threshold:
        return True
    else:
        return False

def random_3d_fiducials(nii_array):
    # Example: Define surface criteria (e.g., thresholding)
    surface_threshold = 0.5  # Example threshold value for surface definition

# Function to check if a point is within the defined surface

    # Number of random points you want to generate
    num_points = 5

    # Generate random coordinates within the array bounds
    max_coords = np.array(nii_array.shape) - 1  # Maximum coordinates in each dimension
    random_coords = np.random.randint(0, max_coords, size=(num_points, 3))

    # Filter random points to keep only those within the surface and with high values
    valid_points = []
    for coord in random_coords:
        if is_within_surface(nii_array, coord, surface_threshold):
            valid_points.append(coord)

def get_3d_fiducials(specimen):
    fiducials = []
    for landmark in specimen["vol-landmarks"]:
        pt_3d = specimen["vol-landmarks"][landmark][:]
        pt_3d = torch.from_numpy(pt_3d)
        fiducials.append(pt_3d)
    return torch.stack(fiducials, dim=0).permute(2, 0, 1)

from torchvision.transforms.functional import center_crop, gaussian_blur


def preprocess(img, size=None, initial_energy=torch.tensor(65487.0)):
    """
    Recover the line integral: $L[i,j] = \log I_0 - \log I_f[i,j]$

    (1) Remove edge due to collimator
    (2) Smooth the image to make less noisy
    (3) Subtract the log initial energy for each ray
    (4) Recover the line integral image
    (5) Rescale image to [0, 1]
    """
    img = center_crop(img, (1436, 1436))
    img = gaussian_blur(img, (5, 5), sigma=1.0)
    img = initial_energy.log() - img.log()
    img = (img - img.min()) / (img.max() - img.min())
    return img

from .calibration import RigidTransform, convert


@beartype
def get_random_offset(batch_size: int, device) -> RigidTransform:
    # may need to change this according to the amount of deviation in camera pose for our data
    r1 = torch.distributions.Normal(0, 0.7).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.7).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 0.7).sample((batch_size,))

    # r1 = -2*torch.ones(batch_size)
    # r2 =  -6*torch.ones(batch_size)

    # r3 = -2.2*torch.ones(batch_size)
    # print("Any one fucking there? - kneefit.py")

    t1 = torch.distributions.Normal(-5, 5).sample((batch_size,))
    t2 = torch.distributions.Normal(-20, 25).sample((batch_size,))
    t3 = torch.distributions.Normal(-25, 25).sample((batch_size,))
    # t1 = torch.zeros((batch_size,))
    # t2 = torch.zeros((batch_size,))
    # t3 = torch.zeros((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        [log_R_vee, log_t_vee],
        "se3_log_map",
        "se3_exp_map",
    )
from torchvision.transforms import Compose, Lambda, Normalize, Resize


class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.transforms = Compose(
            [
                Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)
