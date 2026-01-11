import glob
import os

import seaborn as sns
import cv2
import numpy as np
from matplotlib import pyplot as plt
import onnxruntime as ort

from nerovision.nv_enums import ImageFormat
from nerovision.vision.io import NeroVisionIO
from nerovision.dl.auto_encoding.ad.AEAD import AnomalyDetectorAE


def preprocess_rgb_image(im: np.ndarray, im_h: int, im_w: int) -> np.ndarray:
    resized = cv2.resize(im, (im_w, im_h))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # image: H x W x C, values in [0, 255]
    preprocessed = resized.astype(np.float32) / 255.0
    preprocessed = (preprocessed - mean) / std

    return preprocessed


def revert_normalization(im: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # image_norm: normalized image
    image = (im * std + mean) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main():
    dataset_path = r'C:\dev\autoencoding\data\transistor\test'
    model_path = r'C:\dev\autoencoding\models\model.onnx'
    im_h = 224
    im_w = 224
    im_paths = glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
    anomalous_paths = list(filter(lambda p: 'good' not in p, im_paths))
    good_paths = list(filter(lambda p: 'good' in p, im_paths))

    ort_sess = ort.InferenceSession(model_path)
    anomaly_detector = AnomalyDetectorAE(ort_sess)
    good_scores = []
    anomalous_scores = []

    for im_p in good_paths:
        im = NeroVisionIO.load_im_from_disk(im_p, ImageFormat.rgb)
        anomaly_score = anomaly_detector.calculate_mse_anomaly_score(im, im_h, im_w, ImageFormat.rgb)
        good_scores.append(anomaly_score)

    for im_p in anomalous_paths:
        im = NeroVisionIO.load_im_from_disk(im_p, ImageFormat.rgb)
        anomaly_score = anomaly_detector.calculate_mse_anomaly_score(im, im_h, im_w, ImageFormat.rgb)
        anomalous_scores.append(anomaly_score)

    sns.set_style("darkgrid")
    sns.kdeplot(good_scores, label="Good", fill=True)
    sns.kdeplot(anomalous_scores, label="Anomalous", fill=True)
    plt.xlabel("Anomaly score")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
