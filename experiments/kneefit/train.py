from pathlib import Path

import submitit
import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
import ipdb
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('C:/Users/Public Admin/Desktop/Gitlab/kneedeeppose')

from diffpose.calibration import RigidTransform, perspective_projection
from diffpose.kneefit import KneeFitDataset, Transforms, get_random_offset, get_volume_data
from diffpose.metrics import DoubleGeodesic, GeodesicSE3
from diffpose.registration import PoseRegressor
import tifffile as tiff

# def load(id_number, height, device):
#     specimen = KneeFitDataset(id_number)
#     isocenter_pose = specimen.isocenter_pose.to(device)

#     # Take the height of xray sample and divide by image dimensions required by ResNet
#     subsample = 1000 / height
#     #delx to be retreived from combined csv file
#     # delx is different for different samples, below hard-coded for SUB2
#     delx = 0.2876 * subsample
#     drr = DRR(
#         specimen.volume,
#         specimen.spacing,
#         specimen.focal_len / 2,
#         height,
#         delx,
#         x0=specimen.x0,
#         y0=specimen.y0,
#         reverse_x_axis=True,
#     ).to(device)
#     transforms = Transforms(height)

#     return specimen, isocenter_pose, transforms, drr


def load(id_number, height, device):
    volume, affine = get_volume_data(r"D:\kneefit_model\SUBN_02_Femur_RE_Volume.nii")
    spacing = np.array([0.390625, 0.390625, 0.800018])
    isocenter_rot = torch.tensor([[torch.pi / 2, 0.0, -torch.pi / 2]])
    isocenter_xyz = torch.tensor(volume.shape) * spacing / 2
    isocenter_xyz = isocenter_xyz.unsqueeze(0)
    isocenter_pose = RigidTransform(
        isocenter_rot, isocenter_xyz, "euler_angles", "ZYX"
    )
    isocenter_pose = isocenter_pose.to(device)

    focal_len = 972.2564

    # Take the height of xray sample and divide by image dimensions required by ResNet
    subsample = 1000 / height
    # delx to be retreived from combined csv file
    # delx is different for different samples, below hard-coded for SUB2
    delx = 0.2876 * subsample
    x0 = 5.7096
    y0 = 5.8591
    drr = DRR(
        volume,
        spacing,
        focal_len / 2,
        height,
        delx,
        x0=x0,
        y0=y0,
        reverse_x_axis=False,
    ).to(device)
    transforms = Transforms(height)

    return isocenter_pose, transforms, drr

def visualize( drr, pose , device):
    
    pred_xray = drr(None, None, None, pose=pose.to(device))
    xray = pred_xray[0,:,:,:]
    # print("PRINTING SHIT HERE")
    # print(xray.shape)
    # print(pose.get_translation())
    # print(pose.get_rotation())
    plt.figure(constrained_layout=True)
    plt.subplot(121)
    plt.title("DRR")
    plt.imshow(xray.squeeze().cpu().numpy(), cmap="gray")
    plt.show()

def train(
    id_number,
    model,
    optimizer,
    scheduler,
    drr,
    transforms,
    isocenter_pose,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
):
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)

    best_loss = torch.inf

    model.train()
    #  visualization part
    offset = get_random_offset(batch_size, device)
    pose = isocenter_pose.compose(offset)

    # visualize(drr=drr, pose=pose,device=device)
    for epoch in range(n_epochs + 1):
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            # bone_attenuation is chosen randomly from a uniform distribution
            contrast = contrast_distribution.sample().item()
            # generate random pose using normal distribution
            offset = get_random_offset(batch_size, device)
            pose = isocenter_pose.compose(offset)
            img = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img = transforms(img)
            
            img = img.float()

            pred_offset = model(img)
            pred_pose = isocenter_pose.compose(pred_offset)
            pred_img = drr(None, None, None, pose=pred_pose)
            pred_img = transforms(pred_img)
            pred_img = pred_img.float()

            ncc = metric(pred_img, img)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)
            if loss.isnan().any():
                print("Aaaaaaand we've crashed...")
                print(ncc)
                print(log_geodesic)
                print(geodesic_rot)
                print(geodesic_xyz)
                print(double_geodesic)
                print(pose.get_matrix())
                print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "height": drr.detector.height,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose": pose.get_matrix().cpu(),
                        "pred_pose": pred_pose.get_matrix().cpu(),
                        "img": img.cpu(),
                        "pred_img": pred_img.cpu()
                        **model_params,
                    },
                    f"checkpoints/specimen_{id_number:02d}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                geodesic_rot=geodesic_rot.mean().item(),
                geodesic_xyz=geodesic_xyz.mean().item(),
                geodesic_dou=double_geodesic.mean().item(),
                geodesic_se3=log_geodesic.mean().item(),
                loss=loss.mean().item(),
                ncc=ncc.mean().item(),
            )

            prev_pose = pose
            prev_pred_pose = pred_pose

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{id_number:02d}_best.ckpt",
            )

        if epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"checkpoints/specimen_{id_number:02d}_epoch{epoch:03d}.ckpt",
            )


def main(
    id_number,
    height=256,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=4,
    n_epochs=5,
    n_batches_per_epoch=10,
):
    id_number = int(id_number)

    device = torch.device("cuda")
    isocenter_pose, transforms, drr = load(id_number, height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    Path("checkpoints").mkdir(exist_ok=True)

    train(
        id_number,
        model,
        optimizer,
        scheduler,
        drr,
        transforms,
        isocenter_pose,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
    )

    xray_path = "D:\kneefit_images_full_resolution\C_SUBN_02_dkb_01_002.tif"
    

    # Read the image
    xray_img = tiff.imread('path/to/your/image.tif')

    
    pred_offset = model(xray)
    pred_pose = isocenter_pose.compose(pred_offset)
    predicted_pose = RigidTransform(pred_pose)

    visualize(drr=drr, pose= predicted_pose, device=device)



if __name__ == "__main__":
    # id_numbers = [1, 2, 3, 4, 5, 6]
    # Path("checkpoints").mkdir(exist_ok=True)

    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(
    #     name="deepfluoro",
    #     gpus_per_node=1,
    #     mem_gb=43.5,
    #     slurm_array_parallelism=len(id_numbers),
    #     slurm_partition="A6000",
    #     slurm_exclude="sumac,fennel",
    #     timeout_min=10_000,
    # )
    # jobs = executor.map_array(main, id_numbers)

    main(1)