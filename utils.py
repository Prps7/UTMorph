import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)],indexing='ij')
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

def reverse_warp(x, disp, interp_mode="bilinear"):
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)], indexing='ij')
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] - disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors

def save_flow(target,source,target_seg,source_seg,disp_pred,warped_source,warped_source_seg,save_path):
    save_path = save_path
    ndim = target.ndim - 2
    size = target.size()[2:]
    disp_pred = disp_pred.type_as(target)
    disp = normalise_disp(disp_pred)
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)],indexing='ij')
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)
    op_flow = warped_grid.permute(0,3,1,2)[0,:,:,:]
    op_flow = op_flow.cpu().detach().numpy()

    warped_source = warped_source.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    source = source.cpu().detach().numpy()
    warped_source_seg = warped_source_seg.cpu().detach().numpy()
    source_seg = source_seg.cpu().detach().numpy()
    target_seg = target_seg.cpu().detach().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), layout='constrained')

    # 绘制子图
    axs[0, 0].imshow(target[0,0, :, :], cmap='gray', vmin=0, vmax=255)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(warped_source[0,0, :, :], cmap='gray', vmin=0, vmax=255)
    axs[0, 1].axis('off')

    axs[0, 2].imshow(source[0,0, :, :], cmap='gray', vmin=0, vmax=255)
    axs[0, 2].axis('off')

    # 绘制光流图
    ax = axs[1, 0]
    interval = 7
    for i in range(0, op_flow.shape[1] - 1, interval):
        ax.plot(op_flow[0, i, :], op_flow[1, i, :], c='k', lw=1)

    # 绘制垂直线
    for i in range(0, op_flow.shape[2] - 1, interval):
        ax.plot(op_flow[0, :, i], op_flow[1, :, i], c='k', lw=1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    axs[1, 1].imshow(abs(warped_source[0, 0, :, :] - source[0, 0, :, :]), cmap='gray', vmin=0, vmax=255)
    axs[1, 1].axis('off')

    axs[1, 2].imshow(abs(target[0,0, :, :] - source[0,0, :, :]), cmap='gray', vmin=0, vmax=255)
    axs[1, 2].axis('off')

    axs[2, 0].imshow(target_seg[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axs[2, 0].set_title('Target Segmentation')
    axs[2, 0].axis('off')

    # Display warped source segmentation
    axs[2, 1].imshow(warped_source_seg[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axs[2, 1].set_title('Warped Source Segmentation')
    axs[2, 1].axis('off')

    # Display target prediction
    axs[2, 2].imshow(source_seg[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
    axs[2, 2].set_title('Source Segmentation')
    axs[2, 2].axis('off')

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
