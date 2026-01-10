import pandas as pd

from nerovision.dl.callbacks.base import TrainingCallback


class CSVLogger(TrainingCallback):
    """
    csv logger which writes every epoch a .csv file to the given filepath in order to track the model loss and
    assigned metrics in the trainer logs.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def on_train_begin(self, trainer):
        pass

    def on_train_finished(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        df = pd.DataFrame(trainer.train_logs)
        df.to_csv(self.filepath, index=False)

    def on_epoch_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_val_start(self, trainer):
        pass