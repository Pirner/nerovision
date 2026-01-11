import albumentations as A
from albumentations.pytorch import ToTensorV2

from nerovision.nv_enums import ImageFormat


class DLTransformation(object):
    @staticmethod
    def get_transformations(im_h: int, im_w: int, image_format: ImageFormat):
        """
        a base set of image transformations used for deep learning algorithms
        :param im_h: height of the image for resizing
        :param im_w: width of the image for resizing
        :param image_format: which format the image will have
        :return:
        """
        if image_format == ImageFormat.rgb:
            normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        elif image_format == ImageFormat.gray_16 or image_format == ImageFormat.gray_8:
            normalize = A.Normalize(mean=0.5, std=0.5)
        else:
            raise ValueError('Unsupported image format: {}'.format(image_format))

        t_train = A.Compose(
            [
                # A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.2),
                A.RandomCrop(height=im_h * 2, width=im_w * 2, p=0.2),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.HorizontalFlip(p=0.2),
                A.Affine(p=0.2, scale=0.8, shear=5, translate_percent=0.1, rotate=20),
                A.Blur(blur_limit=3, p=0.1),
                A.OpticalDistortion(p=0.1),
                A.GridDistortion(p=0.1),
                A.HueSaturationValue(p=0.1),
                A.Resize(height=im_h, width=im_w, p=1),
                normalize,
                ToTensorV2(),
            ]
        )
        t_val = A.Compose(
            [
                A.Resize(height=im_h, width=im_w, p=1),
                normalize,
                ToTensorV2(),
            ]
        )
        return t_train, t_val
