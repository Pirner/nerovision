import torch

from nerovision.dl.callbacks.base import TrainingCallback


class Checkpointer(TrainingCallback):
    """
    this callback is made to store the checkpoint of the model
    """
    def __init__(self, model_path: str, im_h: int, im_w: int, n_channels: int, device='cuda'):
        """
        :param model_path: path to where store the model
        :param im_h: image width for the sample tensor for saving the model
        :param im_w: image width for the sample tensor for saving the model
        :param n_channels: number of channels for the sample tensor for saving the model
        :param device: on which device the model is running
        """
        self.model_path = model_path
        self.im_h = im_h
        self.im_w = im_w
        self.n_channels = n_channels
        self.device = device

    def on_train_begin(self, trainer):
        pass

    def on_train_finished(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        example_inputs = torch.randn(1, self.n_channels, self.im_h, self.im_w)
        example_inputs = example_inputs.to(self.device)
        torch.onnx.export(
            trainer.model,
            example_inputs,
            self.model_path,
            export_params=True,  # store trained weights
            opset_version=17,  # ONNX opset version
            do_constant_folding=True,  # optimize constants
            input_names=["input"],
            output_names=["output"]
        )

    def on_epoch_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_val_start(self, trainer):
        pass