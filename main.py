import torch
from torch import nn, optim

from app.data import get_datasets, get_dataloaders, get_test_sample
from app.model import NeuralNetwork, save_model, load_model
from app.train import fit
from app.utils import get_device, default_classes, nvtx_range


def main():
    training_data, test_data = get_datasets()
    train_dataloader, test_dataloader = get_dataloaders(training_data, test_data, batch_size=64)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    device = get_device()
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    fit(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, epochs=5)
    print("Training done!")

    save_model(model, "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load model using helper and perform single-sample prediction
    model2 = load_model("model.pth", device)

    # Single-sample prediction (keep for quick sanity check)
    classes = default_classes()
    model2.eval()
    x, y = get_test_sample(test_data, 0)
    with nvtx_range("inference"):
        with torch.no_grad():
            x = x.to(device)
            pred = model2(x)
            # Handle both batched and unbatched outputs
            out = pred[0] if pred.dim() == 2 else pred
            predicted, actual = classes[out.argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

if __name__ == "__main__":
    main()
