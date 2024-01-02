import numpy as np

NDArray = np.array


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: NDArray) -> NDArray:
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: ... x H x W x C NDArray of an image
        Returns:
            ... x H x W x C ndarray corresponding to image flipped with
            .probability self.p Note: use the provided code to provide
            .randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        img = img.copy()
        if flip_img:
            w = img.shape[1]
            img[..., range(w), :] = img[..., range(w - 1, -1, -1), :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img: NDArray) -> NDArray:
        """ Zero pad and then randomly crop an image.
        Args:
             img: ... x H x W x C NDArray of an image
        Return
            ... x H x W x C NDArray of clipped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        dy, dx = np.random.randint(-self.padding, self.padding + 1, size=2)
        h, w = img.shape[:2]
        # compute the subsection in the image that's still within the frame.
        lx, ux = max(0, dx), min(w, w + dx)
        ly, uy = max(0, dy), min(h, h + dy)
        result = np.zeros_like(img)
        if ux < lx or uy < ly:
            return result
        result[..., ly - dy:uy - dy, lx - dx:ux - dx, :] = img[..., ly:uy, lx:ux, :]
        return result
