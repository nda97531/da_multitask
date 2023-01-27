from typing import List
import torch as tr
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


class TestFlow:
    def __init__(self, device: str, loss_fn: nn.Module):
        self.device = device
        self.loss_fn = loss_fn

    def run_single_task(self, model: nn.Module, dataloader: DataLoader):
        """

        Args:
            model:
            dataloader:

        Returns:

        """
        model = model.eval()

        num_batches = len(dataloader)
        test_loss = 0

        y_true = []
        y_pred = []
        with tr.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = model(x)
                test_loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_pred.append(pred)

        test_loss /= num_batches
        y_true = tr.concatenate(y_true).to('cpu')
        y_pred = tr.concatenate(y_pred).argmax(1).to('cpu')
        print(f'Loss: {test_loss}')
        print(classification_report(y_true, y_pred))

    def run_multitask(self, model: nn.Module, dataloaders: List[DataLoader]):
        """
        
        Args:
            model: 
            dataloaders: 

        Returns:

        """
        model = model.eval()

        # for each task
        for i, dataloader in enumerate(dataloaders):
            if not dataloader:
                continue
            print(f'------------\nTesting task {i}')

            num_batches = len(dataloader)
            task_loss = 0

            y_true = []
            y_pred = []
            with tr.no_grad():
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = model(x, classifier_kwargs={'mask': tr.tensor([i] * len(y))})
                    pred = pred[i]
                    task_loss += self.loss_fn(pred, y).item()
                    y_true.append(y)
                    y_pred.append(pred)

            # calculate log for one task
            task_loss /= num_batches
            y_true = tr.concatenate(y_true).to('cpu')
            y_pred = tr.concatenate(y_pred).argmax(1).to('cpu')
            print(f'Loss: {task_loss}')
            print(classification_report(y_true, y_pred))
