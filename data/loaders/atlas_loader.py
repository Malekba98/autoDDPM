import glob

from transforms.preprocessing import *
import glob
from data.loaders.ixi_loader import IXILoader

from dl_utils import get_data_from_csv

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
        return (
            self.im_t(self.files[idx]),
            *self.get_label(idx))