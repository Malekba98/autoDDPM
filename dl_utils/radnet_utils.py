import torch
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def get_features(image,model):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = model.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image


def compute_fid(model, subset1, subset2):
    synth_features = []
    real_features = []

    real_eval_feats = get_features(subset1,model)
    real_features.append(real_eval_feats)

    # Get the features for the synthetic data
    synth_eval_feats = get_features(subset2,model)
    synth_features.append(synth_eval_feats)

    synth_features = torch.vstack(synth_features)
    real_features = torch.vstack(real_features)

    fid = FIDMetric()
    fid_res = fid(synth_features, real_features)
    return fid_res.item()

def compute_ssim(subset_1,patho_masks,subset_2):
    """
    Compute the Structural Similarity Index (SSIM) between two subsets of images, outside the given pathology masks for
    each pair of images"""

    device = subset_1.device
    
    ssim_values = []
    for i in range(subset_1.shape[0]):  # Iterate over batch dimension
        # Apply the pathology mask to both subsets
        subset_1_masked = subset_1[i] * (1 - patho_masks[i])
        subset_2_masked = subset_2[i] * (1 - patho_masks[i])

        # Calculate SSIM for the masked subsets
        ssim = SSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=4)
        ssim_result = ssim(subset_1_masked.unsqueeze(0), subset_2_masked.unsqueeze(0))  # Add batch dimension

        # Store the mean SSIM value
        ssim_values.append(ssim_result.item())

    # Compute mean and standard deviation of SSIM values
    ssim_mean = torch.tensor(ssim_values, device=device).mean()
    ssim_std = torch.tensor(ssim_values, device=device).std()

    return ssim_mean, ssim_std

def compute_msssim(subset_1,subset_2):
    pass


