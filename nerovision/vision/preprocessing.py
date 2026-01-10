import cv2
import numpy as np

from nerovision.nv_enums import ImageFormat


class VisionPreprocessing:
    @staticmethod
    def preprocess_image(image: np.ndarray, im_h: int, im_w: int, im_format: ImageFormat) -> np.ndarray:
        """
        preprocess the image with normalization and resizing
        :param image: image to preprocess
        :param im_h: image height
        :param im_w: image width
        :param im_format: format of the image to preprocess
        :return:
        """
        if im_format == ImageFormat.rgb:
            return VisionPreprocessing.preprocess_rgb_image(image, im_h, im_w)
        elif im_format == ImageFormat.gray_8:
            raise NotImplementedError('ImageFormat {} is not implemented'.format(im_format))
        elif im_format == ImageFormat.gray_16:
            raise NotImplementedError('ImageFormat {} is not implemented'.format(im_format))
        else:
            raise NotImplementedError('unknown image format: {}'.format(im_format))

    @staticmethod
    def preprocess_rgb_image(im: np.ndarray, im_h: int, im_w: int) -> np.ndarray:
        """
        preprocesses an RGB image for normalization and resizing
        :param im: image to preprocess
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        resized = cv2.resize(im, (im_w, im_h))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # image: H x W x C, values in [0, 255]
        preprocessed = resized.astype(np.float32) / 255.0
        preprocessed = (preprocessed - mean) / std

        return preprocessed

    @staticmethod
    def revert_rgb_normalization(im: np.ndarray) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # image_norm: normalized image
        image = (im * std + mean) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
