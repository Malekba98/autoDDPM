import torchvision.transforms as transforms

from core.DataLoader import DefaultDataset
from transforms.preprocessing import *
import glob


from dl_utils import get_data_from_csv


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
                self.RES,
                transforms.ToPILImage(),
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


class AtlasLoader(IXILoader):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
    ):
        self.mask_dir = mask_dir
        if mask_dir is not None:
            if "csv" in mask_dir[0]:
                self.mask_files = get_data_from_csv(mask_dir)
            else:
                self.mask_files = [
                    glob.glob(mask_dir_i + file_type) for mask_dir_i in mask_dir
                ]
        super(AtlasLoader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )

    def get_label(self, idx):
        if self.label_dir is not None:
            patho_mask = self.seg_t(self.label_files[idx])
        if self.mask_dir is not None:
            brain_mask = self.seg_t(self.mask_files[idx])
            # print(mask_label.shape)
        return (patho_mask, brain_mask)

    def __getitem__(self, idx):
        return (self.im_t(self.files[idx]), *self.get_label(idx))


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
        self.mask_dir = mask_dir
        if mask_dir is not None:
            if "csv" in mask_dir[0]:
                self.mask_files = get_data_from_csv(mask_dir)
            else:
                self.mask_files = [
                    glob.glob(mask_dir_i + file_type) for mask_dir_i in mask_dir
                ]

        super(mask_preprocessing_loader, self).__init__(
            data_dir, file_type, label_dir, mask_dir, target_size, test
        )

    def get_label(self, idx):
        if self.label_dir is not None:
            return self.seg_t(self.label_files[idx])
        else:
            return 0

    def __getitem__(self, idx):
        return (
            self.im_t(self.files[idx]),
            self.get_label(idx),
            idx,
            self.files[idx],  # filename
            self.label_files[idx],  # mask_filename
            self.mask_files[idx],  # brain_mask_filename
        )
