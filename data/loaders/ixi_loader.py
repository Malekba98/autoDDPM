import torchvision.transforms as transforms

from core.DataLoader import DefaultDataset
from transforms.preprocessing import *
import logging

class Flip:
    """
    Flip brain

    """

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return torch.tensor((img.astype(np.float32)).copy())


class IXILoader(DefaultDataset):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
    ):
        self.target_size = target_size
        self.RES = transforms.Resize(self.target_size)
        super(IXILoader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )

    def get_image_transform(self):
        default_t = transforms.Compose(
            [
                ReadImage(),
                To01(),  # , Norm98(),
                Pad((1, 1)),  # Flip(), #  Slice(),
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES
                # ,AdjustIntensity()
                ,
                transforms.ToPILImage(),  # transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8,1.2)),
                transforms.ToTensor(),
            ]
        )
        return default_t

    def get_image_transform_test(self):
        default_t_test = transforms.Compose(
            [
                ReadImage(),
                To01(),  # , Norm98()
                Pad((1, 1))
                # Flip(), #  Slice(),
                ,
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES,
            ]
        )
        return default_t_test

    def get_label_transform(self):
        default_t_label = transforms.Compose(
            [
                ReadImage(),
                To01(),
                Pad((1, 1)),
                AddChannelIfNeeded(),
                AssertChannelFirst(),
                self.RES,
            ]
        )  # , Binarize()])
        return default_t_label


class mask_preprocessing_loader(IXILoader):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
    ):
        super(mask_preprocessing_loader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )


    def get_label(self, idx):
        if self.label_dir is not None:
            return self.seg_t(self.label_files[idx])
        else:
            return 0

    def __getitem__(self, idx):
        return self.im_t(self.files[idx]), self.get_label(idx), idx, self.files[idx]