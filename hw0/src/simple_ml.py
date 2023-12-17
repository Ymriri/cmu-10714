from functools import total_ordering
import struct
import numpy as np
import gzip

from io import BufferedIOBase


# try:
#     from simple_ml_ext import *
# except:
#     pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    return x + y


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.GzipFile(filename=image_filename, mode='rb') as f:
        images = parse_iamge_file(f)

    with gzip.GzipFile(filename=label_filename, mode='rb') as f:
        labels = parse_label_file(f)

    X = np.array(images, dtype=np.float32) / 255.
    y = np.array(labels, dtype=np.uint8)

    return X, y


def parse_label_file(io: BufferedIOBase) -> list[int]:
    # TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    #
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    # 0004     32 bit integer  60000            number of items
    # 0008     unsigned byte   ??               label
    # 0009     unsigned byte   ??               label
    # ........
    # xxxx     unsigned byte   ??               label
    # The labels values are 0 to 9.

    magic_number = _parse_i32(io)
    assert magic_number == 2049, magic_number

    count = _parse_i32(io)
    labels = []
    for _ in range(count):
        data = io.read(1)
        label = int.from_bytes(data, 'big')
        labels.append(label)

    return labels


def parse_iamge_file(io: BufferedIOBase) -> list[list[int]]:
    # TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    #
    # [offset] [type]          [value]          [description]
    # 0000     32 bit integer  0x00000803(2051) magic number
    # 0004     32 bit integer  60000            number of images
    # 0008     32 bit integer  28               number of rows
    # 0012     32 bit integer  28               number of columns
    # 0016     unsigned byte   ??               pixel
    # 0017     unsigned byte   ??               pixel
    # ........
    # xxxx     unsigned byte   ??               pixel
    #
    # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means
    # foreground (black).
    #
    magic_number = _parse_i32(io)
    assert magic_number == 2051, magic_number

    count = _parse_i32(io)
    row = _parse_i32(io)
    col = _parse_i32(io)
    size = row * col

    images = []
    for _ in range(count):
        data = io.read(size)
        image = [int.from_bytes([b], 'big') for b in data]
        images.append(image)

    return images


def _parse_i32(io: BufferedIOBase) -> int:
    data = io.read(4)
    return int.from_bytes(data, 'big')


def softmax(Z):
    """Softmax function

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.

    Returns:
        Average softmax loss over the sample.
    """

    A = np.exp(Z)  # num_examples x num_classes
    Z = A / np.sum(A, axis=1, keepdims=True)  # num_examples x num_classes
    return Z


def relu(Z):
    return (Z >= 0) * Z


def onehot(y, num_classes):
    num_samples = y.shape[0]
    I = np.zeros((num_samples, num_classes), dtype=np.float32)
    I[np.arange(num_samples), y] = 1.
    return I


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """

    return np.mean(
        -Z[np.arange(Z.shape[0]), y] + np.log(np.sum(np.exp(Z), axis=1)),
        axis=0,
    )


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """

    total_samples = X.shape[0]
    assert total_samples % batch == 0
    num_classes = y.max() + 1

    for i in range(0, total_samples, batch):
        mini_X = X[i:i + batch]
        mini_y = y[i:i + batch]

        Z = softmax(mini_X @ theta)  # num_examples x num_classes
        I = onehot(mini_y, num_classes)

        d_theta = mini_X.T @ (Z - I)  # input_dim x num_classes
        theta -= lr / batch * d_theta


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    total_examples = X.shape[0]
    num_classes = y.max() + 1
    assert total_examples % batch == 0

    for i in range(0, total_examples, batch):
        mini_X = X[i:i + batch]
        mini_y = y[i:i + batch]

        A1 = mini_X @ W1
        Z1 = relu(A1)
        A2 = Z1 @ W2
        Z2 = softmax(A2)
        I = onehot(mini_y, num_classes)

        dW2 = Z1.T @ (Z2 - I)  # hidden_dim, num_classes

        # single output
        mask = Z1 > 0
        dW1 = mini_X.T @ ((Z2 - I) @ W2.T * mask)

        W1 -= lr / batch * dW1
        W2 -= lr / batch * dW2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print(X_tr[0])
    print(y_tr[0])

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
