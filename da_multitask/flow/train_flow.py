import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import torch as tr
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from da_multitask.flow.torch_callbacks import TorchCallback, CallbackAction


class TrainFlow:
    def __init__(self, model: tr.nn.Module,
                 optimizer: tr.optim.Optimizer,
                 device: str,
                 loss_fn: tr.nn.Module = 'classification_auto',
                 callbacks: List[TorchCallback] = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = self.auto_classification_loss if loss_fn == 'classification_auto' else loss_fn

        self.train_log = []
        self.valid_log = []

        self.callbacks = callbacks

    @staticmethod
    def auto_classification_loss(inp: tr.Tensor, target: tr.Tensor):

        if len(inp.shape) > 1:
            # if multi-class classification
            if inp.shape[1] > 1:
                loss = F.cross_entropy(inp, target)
                return loss
            else:
                inp = inp.squeeze(1)

        # if binary classification
        loss = F.binary_cross_entropy_with_logits(inp, target.float())
        return loss

    def cal_metric(self, y_true, y_pred):
        """
        Calculate F1 score

        Args:
            y_true: 1d array/tensor shape [num samples]
            y_pred: 2d array/tensor shape [num samples, num class], if num class==1, treat this as a 
                probability array for binary classification; if 1d array, treat as categorical

        Returns:
            float: f1 score
        """
        if len(y_pred.shape) > 1:
            # if multi-class classification
            if y_pred.shape[1] > 1:
                return f1_score(y_true, y_pred.argmax(1), average='macro')

            # if binary classification
            else:
                y_pred = (tr.sigmoid(y_pred.squeeze(1)) >= 0.5).long()
        return f1_score(y_true, y_pred, average='macro')

    def train_loop(self, dataloader: DataLoader):
        self.model = self.model.train()

        train_loss = 0
        y_true = []
        y_pred = []

        pbar = tqdm(total=len(dataloader), ncols=0)
        for batch, (x, y) in enumerate(dataloader):
            x = x.float().to(self.device)
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
            y_pred.append(pred.to('cpu'))

            pbar.update(1)
        pbar.close()

        # record epoch log
        train_loss /= len(dataloader)
        metric = self.cal_metric(tr.concatenate(y_true), tr.concatenate(y_pred))
        self.train_log.append({'loss': train_loss, 'metric': metric})
        print(f'Train: {self.train_log[-1]}')

    def multitask_train_loop(self, dataloaders: List[DataLoader]):
        self.model = self.model.train()

        num_iter = min(len(dataloader) for dataloader in dataloaders)
        dataloaders = [iter(dataloader) for dataloader in dataloaders]

        # 2-level lists to store y_true and y_pred; level 1: task no.; level 2: y true/pred values
        y_true = [list() for _ in dataloaders]
        y_pred = [list() for _ in dataloaders]
        train_loss = 0
        for _ in tqdm(range(num_iter), ncols=0):
            # load data
            data = [next(dataloader) for dataloader in dataloaders]
            x, y = tuple(zip(*data))
            x = tr.concatenate(x).float().to(self.device)
            y = tr.concatenate(y).to(self.device)

            # generate task mask: a 1D array containing a task number for each sample
            task_mask = np.concatenate([[task_idx] * len(data_tensor[0]) for task_idx, data_tensor in enumerate(data)])
            task_mask = tr.from_numpy(task_mask)

            # shuffle this batch so that all tasks are distributed evenly
            shuffle_idx = tr.randperm(len(y))
            x = x[shuffle_idx]
            y = y[shuffle_idx]
            task_mask = task_mask[shuffle_idx]

            # Compute prediction
            pred = self.model(x, classifier_kwargs={'mask': task_mask})
            # divide y by mask, the same way pred is divided
            y = tuple((y[task_mask == i]) for i in range(len(data)))
            # compute loss
            loss = sum(self.loss_fn(pred[i], y[i]) for i in range(len(data))) / len(data)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            train_loss += loss.item()
            for task_no in range(len(data)):
                y_true[task_no].append(y[task_no].to('cpu'))
                y_pred[task_no].append(pred[task_no].to('cpu'))

        # record epoch log
        train_loss /= num_iter
        y_true = [tr.concatenate(cls) for cls in y_true]
        y_pred = [tr.concatenate(cls) for cls in y_pred]
        metrics = [self.cal_metric(y_true[i], y_pred[i]) for i in range(len(y_true))]
        self.train_log.append({'loss': train_loss} | {f'metric_task_{i}': metrics[i] for i in range(len(metrics))})
        print(f'Train: {self.train_log[-1]}')

    def valid_loop(self, dataloader: DataLoader):
        self.model = self.model.eval()

        num_batches = len(dataloader)
        valid_loss = 0

        y_true = []
        y_pred = []
        with tr.no_grad():
            for x, y in dataloader:
                x = x.float().to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                valid_loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_pred.append(pred)

        valid_loss /= num_batches
        y_true = tr.concatenate(y_true).to('cpu')
        y_pred = tr.concatenate(y_pred).to('cpu')
        metric = self.cal_metric(y_true, y_pred)

        # record epoch log
        self.valid_log.append({'loss': valid_loss, 'metric': metric})
        print(f'Valid: {self.valid_log[-1]}')

    def multitask_valid_loop(self, dataloaders: List[DataLoader]):
        """

        Args:
            dataloaders:

        Returns:

        """
        self.model = self.model.eval()

        # list of metric values of all tasks
        metrics = [list() for _ in dataloaders]
        valid_loss = 0
        num_task = 0

        # for each task
        for i, dataloader in enumerate(dataloaders):
            if not dataloader:
                continue

            num_batches = len(dataloader)
            task_loss = 0

            y_true = []
            y_pred = []
            with tr.no_grad():
                for x, y in dataloader:
                    x = x.float().to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x, classifier_kwargs={'mask': tr.tensor([i] * len(y))})
                    pred = pred[i]
                    task_loss += self.loss_fn(pred, y).item()
                    y_true.append(y)
                    y_pred.append(pred)

            # calculate log for one task
            task_loss /= num_batches
            y_true = tr.concatenate(y_true).to('cpu')
            y_pred = tr.concatenate(y_pred).to('cpu')
            metric = self.cal_metric(y_true, y_pred)

            # record log for all tasks
            valid_loss += task_loss
            num_task += 1
            metrics[i] = metric

        # record epoch log
        valid_loss /= num_task
        self.valid_log.append({'loss': valid_loss} | {f'metric_task_{i}': metrics[i] for i in range(len(metrics))})
        print(f'Valid: {self.valid_log[-1]}')

    def run_callbacks(self, epoch: int) -> List[CallbackAction]:
        actions = [
            callback.on_epoch_end(epoch, self.model, self.train_log[-1]['loss'], self.valid_log[-1]['loss'])
            for callback in self.callbacks
        ]
        return actions

    def run(self, train_loader: Union[DataLoader, List[DataLoader]], valid_loader: Union[DataLoader, List[DataLoader]],
            num_epochs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            train_loader:
            valid_loader:
            num_epochs:

        Returns:

        """
        assert type(train_loader) is type(valid_loader), 'train_loader and valid_loader must be of the same type'
        if not isinstance(train_loader, DataLoader):
            assert len(train_loader) == len(valid_loader), \
                'number of tasks in train_loader and valid_loader must the same'

        for epoch in range(1, num_epochs + 1):
            print(f"-----------------\nEpoch {epoch}")

            if isinstance(train_loader, DataLoader):
                self.train_loop(train_loader)
                self.valid_loop(valid_loader)
            else:
                self.multitask_train_loop(train_loader)
                self.multitask_valid_loop(valid_loader)

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
