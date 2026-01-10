from typing import List

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.optim import Adam

from nerovision.dl.auto_encoding.Losses import combined_loss
from nerovision.dl.callbacks.base import TrainingCallback


class AETrainer:
    """
    central class for auto-encoder training
    """
    def __init__(self, model: torch.nn.Module, device='cuda'):
        """
        auto encoding trainer, combines training for auto encoding models
        :param model: the model for encoding
        """
        self.model = model
        self.device = device
        self._train_logs = {}
        self.optimizer = None
        self.epoch = 0
        self.reset_trainer()
        self.loss_fn = combined_loss

    def reset_trainer(self):
        self._train_logs = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        self.epoch = 0

    def _run_val_epoch(self, val_loader: DataLoader, callbacks: List[TrainingCallback]):
        """
        run the validation epoch
        :param val_loader: the validation data loader
        :param callbacks: the callbacks to run during training
        :return:
        """
        self.model.eval()
        val_losses = []

        pbar = tqdm(val_loader, desc='Running Validation epoch')

        with torch.no_grad():
            # for batch in tqdm(val_loader, desc='Running Validation epoch'):
            for batch in pbar:
                x, _ = batch
                x = x.to(self.device)
                recon, _ = self.model(x)
                loss = self.loss_fn(recon, x)
                val_losses.append(loss)
                pbar.set_postfix({'batch_loss': loss.item()})

        val_losses = torch.tensor(val_losses)
        avg_train_loss = torch.mean(val_losses)
        self._train_logs['val_loss'].append(avg_train_loss.item())

    def _run_training_epoch(self, train_loader: DataLoader, callbacks: List[TrainingCallback],):
        """
        run the training epoch
        :param train_loader: the training data loader to run the epoch with
        :param callbacks: the callbacks to run during training
        :return:
        """
        self.model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc='Running Training epoch')

        for batch in pbar:
            x, _ = batch
            x = x.to(self.device)
            recon, _ = self.model(x)

            loss = self.loss_fn(recon, x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss)
            pbar.set_postfix({'batch_loss': loss.item()})

        train_losses = torch.tensor(train_losses)
        avg_train_loss = torch.mean(train_losses)
        self._train_logs['train_loss'].append(avg_train_loss.item())

    def train_model(
            self,
            epochs: int,
            train_loader,
            val_loader,
            callbacks: List[TrainingCallback],
            optimizer=None,
    ):
        """
        training the model
        :param epochs: for how many epochs to train the model
        :param train_loader: torch dataloader for training data
        :param val_loader: torch dataloader for validation data
        :param callbacks: the callbacks to run during training
        :param optimizer: the optimizer to use for training
        :return:
        """
        _ = [x.on_train_begin(self) for x in callbacks]
        self.model = self.model.to(self.device)
        # TODO changes these parts
        if not optimizer:
            self.optimizer = Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        for e in range(epochs):
            print('\nrunning epoch: {} of {}'.format(e + 1, epochs))
            _ = [x.on_epoch_start(self) for x in callbacks]
            self._run_training_epoch(train_loader=train_loader, callbacks=callbacks)
            _ = [x.on_train_end(self) for x in callbacks]

            _ = [x.on_val_start(self) for x in callbacks]
            self._run_val_epoch(val_loader=val_loader, callbacks=callbacks)
            self._train_logs['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            _ = [x.on_epoch_end(self) for x in callbacks]
            print('\nfinished epoch: {}, train_loss: {}, val_loss: {}'.format(e + 1, self._train_logs['train_loss'][-1], self._train_logs['val_loss'][-1]))

        _ = [x.on_train_finished(self) for x in callbacks]

    @property
    def train_logs(self):
        return self._train_logs
