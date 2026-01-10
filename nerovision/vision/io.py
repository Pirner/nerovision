import cv2

from nerovision.nv_enums import ImageFormat


class NeroVisionIO:
    @staticmethod
    def load_im_from_disk(im_path: str, image_format: ImageFormat):
        """

        :param im_path:
        :param image_format:
        :return:
        """
        if image_format == ImageFormat.rgb:
            im = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im_rgb
        elif image_format == ImageFormat.gray_8:
            im = cv2.imread(im_path, 0)
            return im
        elif image_format == ImageFormat.gray_16:
            im = cv2.imread(im_path, -1)
            if len(im.shape) != 2:
                raise Exception("found grayscale 16 dimensions unequal to 2, ", im.shape)
            return im
        else:
            raise NotImplementedError('unsupported image format: {}'.format(image_format))
