import gzip
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
from struct import unpack


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        images, labels = parse_mnist(image_filename, label_filename)
        assert len(images) == len(labels)

        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        images = self.images[index]
        new_shape = images.shape[:-1] + (28, 28, 1)
        old_shape = images.shape
        images = np.reshape(
            self.apply_transforms(np.reshape(images, new_shape)), old_shape
        )
        return images, self.labels[index]

    def __len__(self) -> int:
        return self.images.shape[0]


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
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
    with gzip.GzipFile(filename=image_filename, mode="rb") as f:
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
        # Pixels are organized row-wise. Pixel values are 0 to 255. 0 means
        # background (white), 255 means foreground (black).
        #
        magic, num, num_rows, num_cols = unpack(">4i", f.read(16))
        size = num_rows * num_cols
        assert magic == 2051, magic

        X = np.vstack(
            [
                np.array(unpack(f">{size}B", f.read(size)), dtype=np.float32)
                for _ in range(num)
            ]
        )
        X -= X.min()
        X /= X.max()

    with gzip.GzipFile(filename=label_filename, mode="rb") as f:
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
        magic, num = unpack(">2i", f.read(8))
        assert magic == 2049, magic
        y = np.array(unpack(f">{num}B", f.read(num)), dtype=np.uint8)

    return X, y
