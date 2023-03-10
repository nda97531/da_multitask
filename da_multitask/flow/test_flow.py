from typing import List
import torch as tr
import torch.nn as nn
from torch.utils.data import DataLoader

from da_multitask.flow.flow_functions import auto_classification_loss, classification_report, ypred_2_categorical


class TestFlow:
    def __init__(self, device: str, loss_fn: nn.Module = 'classification_auto'):
        self.device = device
        self.loss_fn = auto_classification_loss if loss_fn == 'classification_auto' else loss_fn

    def run_single_task(self, model: nn.Module, dataloader: DataLoader) -> tuple:
        """

        Args:
            model:
            dataloader:

        Returns:
            a 2-element tuple: (y_true, y_pred), both are categorical 1D array
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
        y_pred = ypred_2_categorical(tr.concatenate(y_pred).to('cpu'))
        print(f'Loss: {test_loss}')
        print(classification_report(y_true, y_pred))
        return y_true, y_pred

    def run_multitask(self, model: nn.Module, dataloaders: List[DataLoader]) -> tuple:
        """
        
        Args:
            model: 
            dataloaders: 

        Returns:
            a 2-level tuple:
                level 1: each element is a task
                level 2: (y_true, y_pred), both are categorical 1D array
        """
        model = model.eval()
        result = []

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
            y_pred = ypred_2_categorical(tr.concatenate(y_pred).to('cpu'))
            print(f'Loss: {task_loss}')
            print(classification_report(y_true, y_pred))
            result.append((y_true, y_pred))
        return tuple(result)
