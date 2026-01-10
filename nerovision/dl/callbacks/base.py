from abc import ABC, abstractmethod

class TrainingCallback(ABC):
    """
    the training callback is being used to run manual operations during training.
    Thereby the method always refers to the trainer which is running the training itself, this allows the system
    to have full access in case somebody creates custom methods for training.
    """
    @abstractmethod
    def on_train_begin(self, trainer):
        pass

    @abstractmethod
    def on_train_finished(self, trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer):
        pass

    @abstractmethod
    def on_epoch_start(self, trainer):
        pass

    @abstractmethod
    def on_train_end(self, trainer):
        pass

    @abstractmethod
    def on_val_start(self, trainer):
        pass
