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

