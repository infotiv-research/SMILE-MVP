"""This module defines a class for undistorting images."""
import numpy as np
import cv2

ALPHA = 0
CENTER = True


class Undistorter:
    """Class for undistorting image."""

    def __init__(self, path, new_width=None, new_height=None, K=None, dist=None, image_shape=None):
        """
        Parameters:
                path (str): Path to the yaml file containing the camera parameters
                new_width (int): distorted image input width. (may be different than the size used for calibration)
                new_height (int): distorted image input height. (may be different than the size used for calibration)
        """
        if K is not None and dist is not None and image_shape is not None:
            self.mtx = K
            self.dist = dist
            self.image_shape = image_shape
        else:
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ )#cv2.FILE_STORAGE_READ
            self.mtx = cv_file.getNode("K").mat()
            self.dist = cv_file.getNode("D").mat()
            # self.xi = cv_file.getNode("xi").mat()
            self.image_shape = cv_file.getNode("image_shape").mat()
            cv_file.release()

        image_shape = np.asarray(self.image_shape).reshape(-1)
        if image_shape.size < 2:
            raise ValueError(f"Invalid image_shape: {self.image_shape!r}")
        self.image_shape = image_shape

        self.width = int(self.image_shape[1])
        self.height = int(self.image_shape[0])

        if new_width is not None and new_height is not None:
            ratio1 = new_width / int(self.image_shape[1])
            ratio2 = new_height / int(self.image_shape[0])
            assert (
                ratio1 == ratio2
            ), "new_width and height must have the same aspect ratio as the images used for calibration."
            # print("ratio1", ratio1)
            self.mtx[0:2, :] = self.mtx[0:2, :] * ratio1

            self.width = new_width
            self.height = new_height

        # During undistortion, I always keep the size of the image (thus using self.width and self.height two times in this function call).
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx,
            self.dist,
            (self.width, self.height),
            ALPHA,
            (self.width, self.height),
            CENTER,
        )  # 0 makes sure all pixels are valid, True makes sure principal point is centered
        # print("self.roi", self.roi)
        x, y, w, h = self.roi
        if h == 0 or w == 0:
            raise Exception
        # since the image is cropped in the undistort function
        # I compute the camera matrix for an image that has been cropped here
        self.K = np.copy(self.newcameramtx)
        self.K[0, 2] -= x
        self.K[1, 2] -= y

    def undistort(self, image: np.ndarray):
        """
        Undistort the input image according to the camera parameters.

                Args:
                        image (np.ndarray): Image to undistort. Needs to be of the same size as w and h above
                Returns:
                        Undistorted image of the same size as the input
        """
        if (image.shape[1] != self.width) or (image.shape[0] != self.height):
            raise Exception(
                "Undistorting images of different size than the ones used for calibration is not allowed."
            )
        else:
            dst = cv2.undistort(image, self.mtx, self.dist, None, self.newcameramtx)
            x, y, w, h = self.roi
            dst = dst[y : y + h, x : x + w]
            return dst
