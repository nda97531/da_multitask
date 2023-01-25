import torch as tr


class TorchCallback:
    def on_epoch_end(self, model: tr.nn.Module, train_result: tr.Tensor, valid_result: tr.Tensor):
        raise NotImplementedError()


class TorchModelCheckpoint(TorchCallback):
    def __init__(self, save_path: str, save_best_only=True, save_weights_only=True):
        self.save_path = save_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

        self.current_best_loss = float('inf')

    def on_epoch_end(self, model: tr.nn.Module, train_result: float, valid_result: float):
        new_loss = valid_result
        if new_loss < self.current_best_loss:
            tr.save(model.state_dict() if self.save_weights_only else model,
                    self.save_path.format(new_loss))
            print(f"Model improved from {self.current_best_loss} to {new_loss}. "
                  f"Save model to {self.save_path}.")

            self.current_best_loss = new_loss


class TorchEarlyStop(TorchCallback):
    def __init__(self, patience: int):
        self.patience = patience
        self.epoch_without_improvements = 0
        self.current_best_loss = float('inf')

    def on_epoch_end(self, model: tr.nn.Module, train_result: float, valid_result: float):
        new_loss = valid_result
        if new_loss < self.current_best_loss:
            self.current_best_loss = new_loss
            self.epoch_without_improvements = 0
        else:
            self.epoch_without_improvements += 1

        stop = self.epoch_without_improvements >= self.patience
        if stop:
            print(f"Model does not improve from {self.current_best_loss}. Stopping")
        return stop
