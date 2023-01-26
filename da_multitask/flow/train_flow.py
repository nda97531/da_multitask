import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import torch as tr
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from da_multitask.flow.torch_callbacks import TorchCallback, CallbackAction


class TrainFlow:
    def __init__(self, model: tr.nn.Module, loss_fn: tr.nn.Module,
                 optimizer: tr.optim.Optimizer, device: str, callbacks: List[TorchCallback] = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.train_log = []
        self.valid_log = []
        self.cal_metric = f1_score

        self.callbacks = callbacks

    def train_loop(self, dataloader: DataLoader):
        self.model = self.model.train()

        train_loss = 0
        y_true = []
        y_pred = []

        pbar = tqdm(total=len(dataloader))
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            # Compute prediction and loss
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            train_loss += loss.item()
            y_true.append(y.to('cpu'))
            y_pred.append(pred.argmax(1).to('cpu'))

            pbar.update(1)
        pbar.close()

        # record epoch log
        train_loss /= len(dataloader)
        metric = self.cal_metric(tr.concatenate(y_true), tr.concatenate(y_pred))
        self.train_log.append({'loss': train_loss, 'metric': metric})
        print(f'Train: {self.train_log[-1]}')

    def valid_loop(self, dataloader: DataLoader):
        self.model = self.model.eval()

        num_batches = len(dataloader)
        valid_loss = 0

        y_true = []
        y_pred = []
        with tr.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                valid_loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_pred.append(pred.argmax(1))

        valid_loss /= num_batches
        y_true = tr.concatenate(y_true).to('cpu')
        y_pred = tr.concatenate(y_pred).to('cpu')
        metric = self.cal_metric(y_true, y_pred)

        # record epoch log
        self.valid_log.append({'loss': valid_loss, 'metric': metric})
        print(f'Valid: {self.valid_log[-1]}')

    def multitask_train_loop(self, dataloaders: List[DataLoader]):
        self.model = self.model.train()

        num_iter = min(len(dataloader) for dataloader in dataloaders)
        dataloaders = [iter(dataloader) for dataloader in dataloaders]

        train_loss = 0
        for batch in tqdm(range(num_iter)):
            # load data
            data = [next(dataloader) for dataloader in dataloaders]
            x, y = tuple(zip(*data))
            x = tr.concatenate(x).to(self.device)
            y = tr.concatenate(y).to(self.device)

            # generate task mask
            task_mask = np.concatenate([[task_idx] * len(data_tensor[0]) for task_idx, data_tensor in enumerate(data)])
            task_mask = tr.from_numpy(task_mask)

            # mix this batch so that all tasks are distributed evenly
            mix_idx = tr.randperm(len(y))
            x = x[mix_idx]
            y = y[mix_idx]
            task_mask = task_mask[mix_idx]

            # Compute prediction
            pred = self.model(x, task_mask)
            # divide y by mask, the same way pred is divided
            y = tuple((y[task_mask == i]) for i in range(len(data)))
            # compute loss
            loss = sum(self.loss_fn(pred[i], y[i]) for i in range(len(data)))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            train_loss += loss.item()

        # record epoch log
        train_loss /= num_iter
        self.train_log.append({'loss': train_loss})
        print(f'Train: {self.train_log[-1]}')

    def run_callbacks(self, epoch: int) -> List[CallbackAction]:
        actions = [
            callback.on_epoch_end(epoch, self.model, self.train_log[-1]['loss'], self.valid_log[-1]['loss'])
            for callback in self.callbacks
        ]
        return actions

    def run(self, train_loader: Union[DataLoader, List[DataLoader]],
            valid_loader: DataLoader, num_epochs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            train_loader:
            valid_loader:
            num_epochs:

        Returns:

        """
        for epoch in range(1, num_epochs + 1):
            print(f"-----------------\nEpoch {epoch}")
            if isinstance(train_loader, DataLoader):
                self.train_loop(train_loader)
            else:
                self.multitask_train_loop(train_loader)
            self.valid_loop(valid_loader)
            callback_actions = self.run_callbacks(epoch)
            if CallbackAction.STOP_TRAINING in callback_actions:
                break

        train_log = pd.DataFrame(self.train_log)
        valid_log = pd.DataFrame(self.valid_log)
        return train_log, valid_log

# if __name__ == '__main__':
#     import numpy as np
#     from da_multitask.networks.classifier import MultiFCClassifiers
#     from da_multitask.data_generator.classification_data_gen import BasicArrayDataset
#
#     model = MultiFCClassifiers(n_features=1, n_classes=[2, 3, 4])
#     datasets = [
#         BasicArrayDataset({
#             0: np.zeros([2, 1]),
#             1: np.zeros([2, 1]) + 1
#         }),
#         BasicArrayDataset({
#             0: np.zeros([2, 1]) + 10,
#             1: np.zeros([2, 1]) + 11,
#             2: np.zeros([2, 1]) + 12,
#         }),
#         BasicArrayDataset({
#             0: np.zeros([2, 1]) + 20,
#             1: np.zeros([2, 1]) + 21,
#             2: np.zeros([2, 1]) + 22,
#             3: np.zeros([2, 1]) + 23,
#         }),
#     ]
#     loaders = [DataLoader(dataset, batch_size=i + 1, shuffle=True) for i, dataset in enumerate(datasets)]
#
#     multitask_train_loop(
#         dataloaders=loaders,
#         model=model,
#         loss_fn=tr.nn.CrossEntropyLoss(),
#         optimizer=tr.optim.SGD(model.parameters(), lr=1e-2)
#     )
