from enum import Enum

import torch as tr


class CallbackAction(Enum):
    NONE = None
    STOP_TRAINING = 'stop_training'


class TorchCallback:
    def on_epoch_end(self, model: tr.nn.Module, train_result: tr.Tensor, valid_result: tr.Tensor) -> CallbackAction:
        raise NotImplementedError()


class TorchModelCheckpoint(TorchCallback):
    def __init__(self, save_path: str, smaller_better: bool = True, save_best_only=True, save_weights_only=True):
        self.save_path = save_path
        self.smaller_better = smaller_better
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

        self.current_best_result = float('inf')

    def on_epoch_end(self, model: tr.nn.Module, train_result: float, valid_result: float) -> CallbackAction:
        new_result = valid_result

        if (self.smaller_better and (new_result < self.current_best_result)) \
                or ((not self.smaller_better) and (new_result > self.current_best_result)):
            tr.save(model.state_dict() if self.save_weights_only else model, self.save_path.format(new_result))
            print(f"Model improved from {self.current_best_result} to {new_result}. "
                  f"Save model to {self.save_path}.")

            self.current_best_result = new_result

        return CallbackAction.NONE


class TorchEarlyStop(TorchCallback):
    def __init__(self, patience: int, smaller_better: bool = True):
        self.patience = patience
        self.smaller_better = smaller_better
        self.epoch_without_improvements = 0
        self.current_best_result = float('inf')

    def on_epoch_end(self, model: tr.nn.Module, train_result: float, valid_result: float):
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
