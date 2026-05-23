import torch

from app.utils import nvtx_range


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    with nvtx_range("train_epoch"):
        for batch, (X, y) in enumerate(dataloader):
            with nvtx_range("train_batch"):
                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if batch % 100 == 0:
                loss_v, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_v:>7f}  [{current:>5d}/{size:>5d}]")


def test_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with nvtx_range("test_epoch"):
        with torch.no_grad():
            for X, y in dataloader:
                with nvtx_range("test_batch"):
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def fit(train_loader, test_loader, model, loss_fn, optimizer, device, epochs: int = 5):
    with nvtx_range("fit"):
        for t in range(epochs):
            with nvtx_range(f"epoch_{t+1}"):
                print(f"Epoch {t+1}\n-------------------------------")
                train_epoch(train_loader, model, loss_fn, optimizer, device)
                test_epoch(test_loader, model, loss_fn, device)
