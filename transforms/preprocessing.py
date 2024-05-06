import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as transform
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform
from monai.utils.enums import TransformBackends
import torchvision.transforms as transforms


class ReadImage(Transform):
    """
    Transform to read image, see torchvision.io.image.read_image
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, path: str) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if ".npy" in path:
            # return torch.tensor(np.flipud((np.load(path).astype(np.float32)).T).copy())  # Mid Axial slice of MRI brains
            img = np.load(path).astype(np.float32)
            img = (img * 255).astype(np.uint8)
            return torch.tensor(img)
        elif ".jpeg" in path or ".jpg" in path or ".png" in path:
            PIL_image = PIL.Image.open(path)
            tensor_image = torch.squeeze(transform.to_tensor(PIL_image))
            return tensor_image
        elif ".nii.gz" in path:
            import nibabel as nip
            from nibabel.imageglobals import LoggingOutputSuppressor

            with LoggingOutputSuppressor():
                img_obj = nip.load(path)
                img_np = np.array(img_obj.get_fdata(), dtype=np.float32)
                img_t = torch.Tensor(img_np.copy())
                # img_t = torch.Tensor(np.flipud(img_np[:, :, 95].T).copy()) # Mid Axial slice of MRI brains
            return img_t
        elif ".nii" in path:
            import nibabel as nip

            img = nip.load(path)
            return torch.Tensor(np.array(img.get_fdata()))
        elif ".dcm" in path:
            from pydicom import dcmread

            ds = dcmread(path)
            return torch.Tensor(ds.pixel_array)
        elif ".h5" in path:  ## !!! SPECIFIC TO FAST MRI, CHANGE FOR OTHER DATASETS
            import h5py

            f = h5py.File(path, "r")
            img_data = f["reconstruction_rss"][:]  # Fast MRI Specific
            img_data = img_data[:, ::-1, :][0]  # flipped up down
            return torch.tensor(img_data.copy())
        else:
            raise IOError


class Norm98:
    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(Norm98, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        # print(torch.min(img), torch.max(img))
        # print(img.shape)
        q = np.percentile(img, 98)
        # q is a pixel value, below which 98% of the pixels lie
        img = img / q
        img[img > 1] = 1
        # return img/self.max_val
        return img


class To01:
    """
    Convert the input to [0,1] scale

    """

    def __init__(self, max_val=255.0):
        self.max_val = max_val
        super(To01, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if torch.max(img) <= 1.0:
            # print(img.cpu().numpy().shape)
            return img
        # print(img.cpu().numpy().shape)
        if torch.max(img) <= 255.0:
            return img / 255

        return img / 65536


class AdjustIntensity:
    def __init__(self):
        self.values = [1, 1, 1, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
        # self.methods = [0, 1, 2]

    def __call__(self, img):
        value = np.random.choice(self.values)
        # method = np.random.choice(self.methods)
        # if method == 0:
        return torchvision.transforms.functional.adjust_gamma(img, value)


class Binarize:
    def __init__(self, th=0.5):
        self.th = th
        super(Binarize, self).__init__()

    def __call__(self, img):
        img[img > self.th] = 1
        img[img < 1] = 0
        return img


class MinMax:
    """
    Min Max Norm
    """

    def __call__(self, img):
        max = torch.max(img)
        min = torch.min(img)
        img = (img - min) / (max - min)
        return img


class ToRGB:
    """
    Convert the input to an np.ndarray from grayscale to RGB

    """

    def __init__(self, r_val, g_val, b_val):
        self.r_val = r_val
        self.g_val = g_val
        self.b_val = b_val
        super(ToRGB, self).__init__()

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        r = np.multiply(img, self.r_val).astype(np.uint8)
        g = np.multiply(img, self.g_val).astype(np.uint8)
        b = np.multiply(img, self.b_val).astype(np.uint8)

        img_color = np.dstack((r, g, b))
        return img_color


class AddChannelIfNeeded(Transform):
    """
    Adds a 1-length channel dimension to the input image, if input is 2D
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if len(img.shape) == 2:
            # print(f'Added channel: {(img[None,...].shape)}')
            return img[None, ...]
        else:
            return img


class AssertChannelFirst(Transform):
    """
    Assert channel is first and permute otherwise
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        assert (
            len(img.shape) == 3
        ), f"AssertChannelFirst:: Image should have 3 dimensions, instead of {len(img.shape)}"
        if img.shape[0] == img.shape[1] and img.shape[0] != img.shape[2]:
            print(f"Permuted channels {(img.permute(2,0,1)).shape}")
            return img.permute(2, 0, 1)
        elif img.shape[0] > 1:
            return img[0:1, :, :]
        else:
            return img


class Slice(Transform):
    """
    Pad with zeros
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # x = int(320 - img.shape[0] / 2)
        # y = int(320 - img.shape[1] / 2)
        # self.pid = (x, y)
        mid_slice = int(img.shape[0] / 2)
        img_slice = img[mid_slice, :, :]
        return img_slice


class Pad(Transform):
    """
    Pad with zeros
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, pid=(1, 1)):
        self.pid = pid

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = torch.squeeze(img)
        max_dim = max(img.shape[0], img.shape[1])
        x = max_dim - img.shape[0]
        y = max_dim - img.shape[1]

        self.pid = (int(y / 2), y - int(y / 2), int(x / 2), x - int(x / 2))
        pad_val = torch.min(img)
        img_pad = F.pad(img, self.pid, "constant", pad_val)

        return img_pad


class Zoom(Transform):
    """
    Resize 3d volumes
    """

    def __init__(self, input_size):
        self.input_size = input_size
        self.mode = "trilinear" if len(input_size) > 2 else "bilinear"

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if len(img.shape) == 3:
            img = img[None, ...]
        return F.interpolate(img, size=self.input_size, mode=self.mode)[0]


from skimage.exposure import match_histograms
from PIL import Image
import numpy as np


class Harmonize:

    def __init__(self):
        image_reference_path = (
            "/home/malek/autoDDPM/data/MNI152_template/mni_95_slice_template.png"
        )
        image_reference = Image.open(image_reference_path)
        self.image_reference = np.array(image_reference)[:, :, 0]

        pad = Pad()

        self.image_reference = pad(torch.tensor(self.image_reference))
        self.image_reference = self.image_reference.numpy()

    def __call__(self, image):

        image_np = image.numpy()
        # image_np = image_np[0]
        print("shape of image_np", image_np.shape)
        print("shape of image_reference", self.image_reference.shape)
        matched_np = match_histograms(image_np, self.image_reference)
        matched_tensor = torch.from_numpy(matched_np)
        matched_tensor = matched_tensor.view(image.size())
        print("shape of matched_tensor", matched_tensor.shape)
        return matched_tensor


class HarmonizeToMNI:
    def __init__(self):
        image_reference_path = (
            "/home/malek/autoDDPM/data/MNI152_template/mni_95_slice_template.png"
        )

        image_reference = Image.open(image_reference_path)
        self.image_reference = np.array(image_reference)[:, :, 0]
        pad = Pad()
        resize = transforms.Resize([128, 128])

        self.image_reference = pad(torch.tensor(self.image_reference))
        self.image_reference = resize(self.image_reference.unsqueeze(0)).squeeze(0)
        self.image_reference = self.image_reference.numpy()

    def __call__(self, image):

        image_np = image.numpy()
        # image_np = image_np[0]
        print("shape of image_np", image_np.shape)
        print("shape of image_reference", self.image_reference.shape)
        matched_np = match_histograms(image_np, self.image_reference)
        matched_tensor = torch.from_numpy(matched_np)
        matched_tensor = matched_tensor.view(image.size())
        print("shape of matched_tensor", matched_tensor.shape)
        return matched_tensor
