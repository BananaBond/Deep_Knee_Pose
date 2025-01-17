# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/03_registration.ipynb.

# %% auto 0
__all__ = ['PoseRegressor', 'SparseRegistration', 'VectorizedNormalizedCrossCorrelation2d']

# %% ../notebooks/api/03_registration.ipynb 3
import timm
import torch

# %% ../notebooks/api/03_registration.ipynb 5
from .calibration import RigidTransform, convert


class PoseRegressor(torch.nn.Module):
    """
    A PoseRegressor is comprised of a pretrained backbone model that extracts features
    from an input X-ray and two linear layers that decode these features into rotational
    and translational camera pose parameters, respectively.
    """

    def __init__(
        self,
        model_name,
        parameterization,
        denorm_xyz = torch.tensor([1.0,1,1]),
        denorm_rot = torch.tensor([2.1,1.5,0.5]),
        convention=None,
        pretrained=False,
        batch_norm=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]
        self.denorm_xyz = denorm_xyz
        self.denorm_rot = denorm_rot
        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            **kwargs,
        )

        output = self.backbone(torch.randn(1, 1, 256, 256)).shape[-1]
        # self.hidden_xyz_1 = torch.nn.Linear(output, 3)
        # self.hidden_xyz_2 = torch.nn.Linear(128, 32)
        self.xyz_regression = torch.nn.Linear(output, 3)

        # self.hidden_rot_1 =  torch.nn.Linear(output, n_angular_components)
        # self.hidden_rot_2 =  torch.nn.Linear(128, 32)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)
        # self.r_regression = torch.nn.Linear(output, 1)
        # self.o_regression = torch.nn.Linear(output, 1)
        # self.t_regression = torch.nn.Linear(output, 1)


        self.weights = torch.tensor([1/2.1, 1/1.5, 1/0.5], dtype=torch.float).to("cuda")
        # self.activation = torch.nn.LeakyReLU()
        # self.batch_norm_1  = torch.nn.BatchNorm1d(32)
        # self.dropout = torch.nn.Dropout(p=0.5)
        # self.batch_norm_2  = torch.nn.BatchNorm1d(32)
        # self.apply(self.init_weights)
        if not batch_norm:
            self.replace_batchnorms_instancenorm(self.backbone)


    def op_make_sense(self, xyz, rot):
        xyz_log = xyz* self.denorm_xyz
        rot_log = rot * self.denorm_rot

        return convert(
            [rot_log, xyz_log],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

    def forward(self, x):
        x = self.backbone(x)
        
        # h1_rot = (self.hidden_rot_1(x))
        # h2_rot = self.batch_norm_2(self.relu(self.hidden_rot_2(h1_rot)))
        # h1_rot = self.relu(self.hidden_xyz_1(x))
        # h2_rot = self.relu(self.hidden_xyz_2(h1_rot))
        
        rot = self.rot_regression(x)
        # rot = rot * self.weights
        # r = self.r_regression(x)
        # o = self.o_regression(x)
        # t = self.t_regression(x)
        # print(r,o,t)
        # rot = torch.stack([r, o, t], dim=1).squeeze()
        # print(rot)

        
        # h1_xyz = self.hidden_xyz_1(x)
        # h2_xyz = self.batch_norm_2(self.relu(self.hidden_xyz_2(h1_xyz)))
        xyz = self.xyz_regression(x)

        # print(xyz)

        return convert(
            [rot, xyz],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )
        # return xyz, rot

      # hook for the gradients of the activations
    
    def replace_batchnorms_instancenorm(self, node):
        for child_name, child in node.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                setattr(
                    node,
                    child_name,
                    torch.nn.InstanceNorm2d(child.num_features, affine=True, track_running_stats=False),
                )
            else:
                self.replace_batchnorms_instancenorm(child)



# %% ../notebooks/api/03_registration.ipynb 6
N_ANGULAR_COMPONENTS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "se3_log_map": 3,
    "quaternion": 4,
    "rotation_6d": 6,
    "rotation_10d": 10,
    "quaternion_adjugate": 10,
}

# %% ../notebooks/api/03_registration.ipynb 11
from diffdrr.detector import make_xrays
from diffdrr.drr import DRR
from diffdrr.siddon import siddon_raycast

from .calibration import RigidTransform


class SparseRegistration(torch.nn.Module):
    def __init__(
        self,
        drr: DRR,
        pose: RigidTransform,
        parameterization: str,
        convention: str = None,
        features=None,  # Used to compute biased estimate of mNCC
        n_patches: int = None,  # If n_patches is None, render the whole image
        patch_size: int = 13,
    ):
        super().__init__()
        self.drr = drr

        # Parse the input pose
        rotation, translation = convert(
            pose,
            input_parameterization="se3_exp_map",
            output_parameterization=parameterization,
            output_convention=convention,
        )
        self.parameterization = parameterization
        self.convention = convention
        self.rotation = torch.nn.Parameter(rotation)
        self.translation = torch.nn.Parameter(translation)

        # Crop pixels off the edge such that pixels don't fall outside the image
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.patch_radius = self.patch_size // 2 + 1
        self.height = self.drr.detector.height
        self.width = self.drr.detector.width
        self.f_height = self.height - 2 * self.patch_radius
        self.f_width = self.width - 2 * self.patch_radius

        # Define the distribution over patch centers
        if features is None:
            features = torch.ones(
                self.height, self.width, device=self.rotation.device
            ) / (self.height * self.width)
        self.patch_centers = torch.distributions.categorical.Categorical(
            probs=features.squeeze()[
                self.patch_radius : -self.patch_radius,
                self.patch_radius : -self.patch_radius,
            ].flatten()
        )

    def forward(self, n_patches=None, patch_size=None):
        # Parse initial density
        if not hasattr(self.drr, "density"):
            self.drr.set_bone_attenuation_multiplier(
                self.drr.bone_attenuation_multiplier
            )

        if n_patches is not None or patch_size is not None:
            self.n_patches = n_patches
            self.patch_size = patch_size

        # Make the mask for sparse rendering
        if self.n_patches is None:
            mask = torch.ones(
                1,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
        else:
            mask = torch.zeros(
                self.n_patches,
                self.height,
                self.width,
                dtype=torch.bool,
                device=self.rotation.device,
            )
            radius = self.patch_size // 2
            idxs = self.patch_centers.sample(sample_shape=torch.Size([self.n_patches]))
            idxs, jdxs = (
                idxs // self.f_height + self.patch_radius,
                idxs % self.f_width + self.patch_radius,
            )

            idx = torch.arange(-radius, radius + 1, device=self.rotation.device)
            patches = torch.cartesian_prod(idx, idx).expand(self.n_patches, -1, -1)
            patches = patches + torch.stack([idxs, jdxs], dim=-1).unsqueeze(1)
            patches = torch.concat(
                [
                    torch.arange(self.n_patches, device=self.rotation.device)
                    .unsqueeze(-1)
                    .expand(-1, self.patch_size**2)
                    .unsqueeze(-1),
                    patches,
                ],
                dim=-1,
            )
            mask[
                patches[..., 0],
                patches[..., 1],
                patches[..., 2],
            ] = True

        # Get the source and target
        pose = convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )
        source, target = make_xrays(
            pose,
            self.drr.detector.source,
            self.drr.detector.target,
        )

        # Render the sparse image
        target = target[mask.any(dim=0).view(1, -1)]
        img = siddon_raycast(source, target, self.drr.density, self.drr.spacing)
        if self.n_patches is None:
            img = self.drr.reshape_transform(img, batch_size=len(self.rotation))
        return img, mask

    def get_current_pose(self):
        return convert(
            [self.rotation, self.translation],
            input_parameterization=self.parameterization,
            output_parameterization="se3_exp_map",
            input_convention=self.convention,
        )

# %% ../notebooks/api/03_registration.ipynb 13
def preprocess(x, eps=1e-4):
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return (x - 0.3080) / 0.1494


def pred_to_patches(pred_img, mask, n_patches, patch_size):
    return pred_img.expand(-1, n_patches, -1)[..., mask[..., mask.any(dim=0)]].reshape(
        1, n_patches, -1
    )


def img_to_patches(img, mask, n_patches, patch_size):
    return img.expand(-1, n_patches, -1, -1)[..., mask].reshape(1, n_patches, -1)


def mask_to_img(img, mask):
    return img[..., mask.any(dim=0)]


def vector_to_img(pred_img, mask):
    patches = [pred_img]
    filled = torch.zeros(1, 1, *mask[0].shape, device=pred_img.device)
    filled[...] = torch.nan
    for idx in range(len(mask)):
        patch = pred_img[:, mask[idx][mask.any(dim=0)]]
        filled[..., mask[idx]] = patch
        patches.append(patch)
    return filled

# %% ../notebooks/api/03_registration.ipynb 14
class VectorizedNormalizedCrossCorrelation2d(torch.nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, img, pred_img, mask, n_patches, patch_size):
        pred_img = preprocess(pred_img).unsqueeze(0)
        sub_img = mask_to_img(img, mask)
        pred_patches = pred_to_patches(pred_img, mask, n_patches, patch_size)
        img_patches = img_to_patches(img, mask, n_patches, patch_size)

        local_ncc = self.forward_compute(pred_patches, img_patches)
        global_ncc = self.forward_compute(pred_img, sub_img)
        return (local_ncc + global_ncc) / 2

    def forward_compute(self, x1, x2):
        assert x1.shape == x2.shape, "Input images must be the same size"
        x1, x2 = self.norm(x1), self.norm(x2)
        ncc = (x1 * x2).mean(dim=[-1, -2])
        return ncc

    def norm(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0) + self.eps
        std = var.sqrt()
        return (x - mu) / std
