from typing import List

from torch.utils.data import Dataset

from nerovision.nv_enums import ImageFormat
from nerovision.vision.io import NeroVisionIO


class FilepathAEDataset(Dataset):
    """
    this dataset loads image from filepaths, so given a list of strings it starts to load them for autoencoder
    reconstruction purposes
    """
    def __init__(self, filepaths: List[str], transformations, im_format: ImageFormat):
        """
        :param filepaths:
        :param transformations:
        """
        self.transformations = transformations
        self.filepaths = filepaths
        self.im_format = im_format

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        """
        this function returns an item given an index from the dataset.
        the returned value are two tensors of the image transformed
        :param index: which index to load the datapoint from
        :return:
        """
        im = NeroVisionIO.load_im_from_disk(im_path=self.filepaths[index], image_format=self.im_format)
        transformed = self.transformations(image=im)
        transformed_im = transformed["image"]

        return transformed_im, transformed_im
