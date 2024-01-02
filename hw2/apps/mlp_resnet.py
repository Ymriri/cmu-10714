from typing import Optional

import needle as ndl
import needle.nn as nn
import numpy as np
import os

np.random.seed(0)


# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    modules = (
        [
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
        ]
        + [
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ]
        + [nn.Linear(hidden_dim, num_classes)]
    )
    return nn.Sequential(*modules)


def epoch(
    dataloader: ndl.data.DataLoader,
    model: nn.Module,
    opt: Optional[ndl.optim.Optimizer] = None,
) -> (float, float):
    """
    Returns:
        The average error rate, and the average loss.
    """
    np.random.seed(4)
    count = 0
    total_loss = 0.0
    errors = 0.0
    loss_fn = nn.SoftmaxLoss()

    if opt:
        model.train()
        for xs, ys in dataloader:
            m = xs.shape[0]
            logits = model(xs)
            loss = loss_fn(logits, ys)
            total_loss += loss.numpy() * m
            errors += np.sum(logits.numpy().argmax(axis=1) != ys.numpy())
            count += m

            opt.reset_grad()
            loss.backward()
            opt.step()

    else:
        model.eval()
        for xs, ys in dataloader:
            m = xs.shape[0]
            logits = model(xs)
            total_loss += loss_fn(logits, ys).numpy() * m
            errors += np.sum(logits.numpy().argmax(axis=1) != ys.numpy())
            count += m

    return errors / count, total_loss / count


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    dim = 28 * 28
    num_classes = 10

    train_images_file = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_file = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    dataset = ndl.data.MNISTDataset(train_images_file, train_label_file)
    dataloader = ndl.data.DataLoader(dataset, batch_size, shuffle=True)
    model = MLPResNet(dim, hidden_dim, num_classes=num_classes)
    opt = optimizer(model.parameters(), weight_decay=weight_decay, lr=lr)
    for _ in range(epochs):
        errors, loss = epoch(dataloader, model, opt)
    train_errors, train_loss = errors, loss

    test_images_file = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_file = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    dataset = ndl.data.MNISTDataset(test_images_file, test_label_file)
    dataloader = ndl.data.DataLoader(dataset, len(dataset))
    test_errors, test_loss = epoch(dataloader, model)

    return train_errors, train_loss, test_errors, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
