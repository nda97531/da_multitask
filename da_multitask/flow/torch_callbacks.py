from enum import Enum
import os
import torch as tr


class CallbackAction(Enum):
    NONE = None
    STOP_TRAINING = 'stop_training'


class TorchCallback:
    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_result: float, valid_result: float) -> CallbackAction:
        raise NotImplementedError()


class ModelCheckpoint(TorchCallback):
    def __init__(self, num_epochs: int, save_path: str, smaller_better: bool = True, save_best_only=True,
                 save_weights_only=True):
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.smaller_better = smaller_better
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.current_best_result = float('inf')
        self.current_best_epoch = -1

        os.makedirs(os.path.split(self.save_path)[0], exist_ok=True)

    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_result: float, valid_result: float) -> CallbackAction:
        assert epoch != 0, 'Epoch starts at 1'

        new_result = valid_result
        save_path = self.save_path.format(new_result)

        # if save every epoch
        if not self.save_best_only:
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Save model to {save_path}.")

        # if only save improved model
        elif (self.smaller_better and (new_result < self.current_best_result)) \
                or ((not self.smaller_better) and (new_result > self.current_best_result)):
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Model improved from {self.current_best_result} to {new_result}. "
                  f"Save model to {save_path}.")
            self.current_best_result = new_result
            self.current_best_epoch = epoch

        else:
            print(f'Not improved from {self.current_best_result} at epoch {self.current_best_epoch}')

        # save last epoch
        if self.num_epochs == epoch:
            save_path, extension = os.path.splitext(save_path)
            save_path = f'{save_path}_last_epoch{extension}'
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Save last epoch to {save_path}.")

        return CallbackAction.NONE


class EarlyStop(TorchCallback):
    def __init__(self, patience: int, smaller_better: bool = True):
        self.patience = patience
        self.smaller_better = smaller_better
        self.epoch_without_improvements = 0
        self.current_best_result = float('inf')

    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_result: float, valid_result: float):
        new_result = valid_result
        if (self.smaller_better and (new_result < self.current_best_result)) \
                or ((not self.smaller_better) and (new_result > self.current_best_result)):
            self.current_best_result = new_result
            self.epoch_without_improvements = 0
        else:
            self.epoch_without_improvements += 1

        if self.epoch_without_improvements >= self.patience:
            print(f"Model does not improve from {self.current_best_result}. Stopping")
            return CallbackAction.STOP_TRAINING
        return CallbackAction.NONE
