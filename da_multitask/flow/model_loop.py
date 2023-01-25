import torch as tr
from tqdm import tqdm
from sklearn.metrics import classification_report

def train_loop(dataloader, model, loss_fn, optimizer):
    pbar = tqdm(total=len(dataloader))
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    y_true = []
    y_pred = []
    with tr.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_true.append(y)
            y_pred.append(pred.argmax(1))

    test_loss /= num_batches
    y_true = tr.concatenate(y_true).numpy()
    y_pred = tr.concatenate(y_pred).numpy()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    print(classification_report(y_true, y_pred))
