import gzip
import struct

import numpy as np


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
    # 需要把2D转成以1D
    images = read_mnist_images(image_filename)

    labels = read_mnit_labels(label_filename)

    # 把输入的值约束在0-1
    X = np.array(images, dtype=np.float32) / 255.

    # 标签
    y = np.array(labels, dtype=np.uint8)

    return X, y


def read_mnist_images(filename: str):
    with gzip.open(filename, 'rb') as f:
        # 文件头：前16个字节 , 分别获得魔数、图像数量、行数、列数
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # 读取图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows*cols)
        return images


def read_mnit_labels(filename: str):
    with gzip.open(filename, 'rb') as f:
        # 文件头：前8个字节
        magic_number, num_items = struct.unpack(">II", f.read(8))
        # 读取标签数据
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data


def softmax(Z):
    """Softmax function

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.

    Returns:
        Average softmax loss over the sample.
    """
    # 有时候会减去最大的分数
    A = np.exp(Z)  # num_examples x num_classes
    Z = A / np.sum(A, axis=1, keepdims=True)  # num_examples x num_classes
    return Z


def relu(Z):
    return (Z >= 0) * Z


def onehot(y, num_classes):
    """

    Args:
        y: 输入的标签
        num_classes:

    Returns:
       输出以标签为索引的onehot矩阵
    """
    num_samples = y.shape[0]
    # new一个全0的矩阵
    I = np.zeros((num_samples, num_classes), dtype=np.float32)
    # 生成索引，把y对应的位置设置为1
    I[np.arange(num_samples), y] = 1.
    return I

def cross_entropy_loss(Z, y):
    """
    Compute the cross-entropy loss for the given predictions Z and labels y.
    Args:
        Z:
        y:

    Returns:

    """
    return np.mean(-np.log(Z[np.arange(y.shape[0]), y]))

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
    soft_probs = softmax(Z)

    loss = cross_entropy_loss(soft_probs, y)
    return loss


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
        # 依次拿到批次
        mini_X = X[i:i + batch]
        mini_y = y[i:i + batch]
        # 计算softmax ，(bath, num_classes)
        Z = softmax(mini_X @ theta)  # num_examples x num_classes
        I = onehot(mini_y, num_classes)
        # 计算出每个feature的梯度，根据误差计算，（n_features,num_classes）
        d_theta = mini_X.T @ (Z - I)  # input_dim x num_classes
        # 更新theta
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
        # 2层神经网络
        A1 = mini_X @ W1
        Z1 = relu(A1)
        A2 = Z1 @ W2
        Z2 = softmax(A2)

        I = onehot(mini_y, num_classes)

        dW2 = Z1.T @ (Z2 - I)  # hidden_dim, num_classes

        # single output，因为经过了激活层
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
    # 这个就是权重参数
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
    # 两个随机的权重
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        # 直接简单的实现2层网络
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        # 计算测试值
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    # X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
    #                          "data/train-labels-idx1-ubyte.gz")
    # X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
    #                          "data/t10k-labels-idx1-ubyte.gz")
    # X, y = parse_mnist("../data/train-images-idx3-ubyte.gz",
    #                    "../data/train-labels-idx1-ubyte.gz")
    # print(X.shape)
    # X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
    #                          "../data/train-labels-idx1-ubyte.gz")
    # X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
    #                          "../data/t10k-labels-idx1-ubyte.gz")
    #
    # train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.2, batch=100)

    X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr=0.2)
    #
    # print(X_tr[0])
    # print(y_tr[0])
    #
    # print("Training softmax regression")
    # train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)
    #
    # print("\nTraining two layer neural network w/ 100 hidden units")
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
