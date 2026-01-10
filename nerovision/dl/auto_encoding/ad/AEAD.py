import numpy as np

from nerovision.nv_enums import ImageFormat
from nerovision.vision.preprocessing import VisionPreprocessing


class AnomalyDetectorAE:
    """
    this class is being used to perform anomaly detection and calibrate an anomaly detector based on
    an autoencoder.
    There are multiple ways to perform anomaly detection on images.
    The autoencoder needs to return 2 tensors (reconstruction, latent_space) in order to be compatible with this class.
    """
    def __init__(self, ae):
        """
        :param ae: autoencoder instance
        """
        self.ae = ae

    def _infer_im(self, preprocessed):
        """
        run the image through the model after preprocessing
        :param preprocessed:
        :return:
        """
        preprocessed = preprocessed.transpose((2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0).astype(np.float32)
        outputs = self.ae.run(None, {'input': preprocessed})
        recon = outputs[0][0]
        latent_vector = outputs[1][0]
        return recon, latent_vector

    def calculate_mse_anomaly_score(self, im_data: np.ndarray, im_h: int, im_w: int, im_format: ImageFormat = ImageFormat.rgb) -> float:
        """
        calculates the mean squared error anomaly score between two images.
        :param im_data: the image data to calculate the anomaly score for
        :param im_h: the height of the model
        :param im_w: the width of the model
        :param im_format: which image format to calculate the anomaly score for
        :return:
        """
        # preprocess image
        preprocessed = VisionPreprocessing.preprocess_image(im_data, im_h, im_w, im_format)
        # run image through the model
        recon, _ = self._infer_im(preprocessed)
        recon = np.transpose(recon, (1, 2, 0))
        # calculate mse score
        mse = np.mean((recon - preprocessed) ** 2)
        return float(mse)
