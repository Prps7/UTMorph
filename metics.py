import SimpleITK as sitk
import torch
import numpy as np
from monai.metrics import compute_dice


def batch_hausdorff_distance(y_true, y_pred, percentile=95):
    assert y_true.shape == y_pred.shape, "Input shape requirements must be satisfied."

    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

    if y_true.ndim == 3:
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]

    batch_size, num_channels = y_true.shape[:2]
    hd_list = []

    for b in range(batch_size):
        for c in range(num_channels):
            true_mask = y_true[b, c] > 0
            pred_mask = y_pred[b, c] > 0

            if not np.any(true_mask) or not np.any(pred_mask):
                hd_list.append(np.nan)
                continue

            coords_true = np.argwhere(true_mask)
            coords_pred = np.argwhere(pred_mask)

            d_matrix = np.sqrt(
                ((coords_true[:, None] - coords_pred) ** 2).sum(axis=2)
            )

            hd = max(
                np.percentile(np.min(d_matrix, axis=1), percentile),
                np.percentile(np.min(d_matrix, axis=0), percentile)
            )
            hd_list.append(hd)

    return np.nanmean(np.array(hd_list).reshape(batch_size, num_channels), axis=0)

def calculate_metrics(pred, target):
    assert pred.dim() == 4 and target.dim() == 4, "The input tensor must be in BCHW format."
    assert pred.size() == target.size(), "Input shape requirements must be satisfied."

    dice = compute_dice(y_pred=pred, y=target)[:,0]
    dice = dice.cpu().detach().numpy()

    diff = pred - target
    msd = (diff ** 2).mean(dim=(2, 3)).mean()

    hd95 = batch_hausdorff_distance(pred.cpu().detach().numpy() > 0, target.cpu().detach().numpy() > 0)

    return dice, msd.item(), hd95

def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    folding_ratio = []
    mag_grad_jac_det = []
    for n in range(disp.shape[0]):
        disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        jac_det_n = calculate_jacobian_det(disp_n)
        folding_ratio += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_jac_det += [np.abs(np.gradient(jac_det_n)).mean()]
    return np.mean(folding_ratio), np.mean(mag_grad_jac_det)

def calculate_jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.adarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp, isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det

def compute_distances(coords1, coords2):
    """
    Compute all pairwise Euclidean distances between two sets of points.

    :param coords1: Coordinates of points in the first set.
    :param coords2: Coordinates of points in the second set.
    :return: A matrix containing the pairwise distances.
    """
    # Compute the difference matrix
    diff_matrix = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    # Compute the squared Euclidean distances
    dist_squared = np.sum(diff_matrix ** 2, axis=-1)
    # Take square root to get the Euclidean distances
    distances = np.sqrt(dist_squared)
    return distances


def hausdorff_distance(y_true, y_pred, percentile=95):
    """
    Calculate the Hausdorff distance between two binary tensors without using scipy.
    """
    assert y_true.shape == y_pred.shape, "The shapes of y_true and y_pred must match."

    # Ensure that both inputs are boolean tensors
    y_true_bool = y_true > 0
    y_pred_bool = y_pred > 0

    # Find the coordinates of the points
    coords_true = np.argwhere(y_true_bool)
    coords_pred = np.argwhere(y_pred_bool)

    # Check if either set is empty
    if coords_true.size == 0 or coords_pred.size == 0:
        raise ValueError("At least one of the input tensors does not contain any non-zero elements.")

    # Compute all pairwise distances
    distances = compute_distances(coords_true, coords_pred)

    # Compute the Hausdorff distance as the maximum of the two directions
    hausdorff_distance = max(
        np.percentile(np.min(distances, axis=0), percentile),
        np.percentile(np.min(distances, axis=1), percentile)
    )

    return hausdorff_distance


# 示例使用
if __name__ == "__main__":
    pred = torch.rand(4, 1, 128, 128, device='cuda')
    target = torch.rand(4, 1, 128, 128, device='cuda')

    dice, msd, hd = calculate_metrics(pred, target)
    print(f"Dice: {dice:.4f}, MSD: {msd:.4f}, HD: {hd:.4f}")

