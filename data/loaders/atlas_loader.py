import glob

from transforms.preprocessing import *
import glob
from data.loaders.ixi_loader import IXILoader

from dl_utils import get_data_from_csv
from dl_utils.mask_utils import dilate_mask


class AtlasLoader(IXILoader):
    def __init__(
        self,
        data_dir,
        file_type="",
        label_dir=None,
        mask_dir=None,
        target_size=(256, 256),
        test=False,
        dilation_kernel=3,
    ):
        self.dilation_kernel = dilation_kernel
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
        patho_mask = 0
        brain_mask = 0

        if self.label_dir is not None:
            patho_mask = self.seg_t(self.label_files[idx])
        if self.mask_dir is not None:
            brain_mask = self.seg_t(self.mask_files[idx])
        return (patho_mask, brain_mask)

    def get_dilated_mask(self, idx):
        (patho_mask, brain_mask) = self.get_label(idx)

        dilated_patho_mask = dilate_mask(patho_mask, kernel=self.dilation_kernel)
        dilated_patho_mask[brain_mask == 0] = 0
        return dilated_patho_mask

    def __getitem__(self, idx):
        return (
            self.im_t(self.files[idx]),
            *self.get_label(idx),
            self.get_dilated_mask(idx),
        )
