"""This module provides functionality to transform ATR detection to factory coordinate frame."""
# TODO an implementation that uses aruco.pose
import numpy as np
from typing import Dict, Tuple, List
from undistorter import Undistorter

class CamToFactory:
    """This class provides functionality for transforming ATR detection to 3D."""

    def __init__(
        self,
        K, dist, image_shape,
        rot_mat, t_vec,
        # image_shape: Optional[Tuple[int, int]],
        new_width: int,
        new_height: int,
        atr_height: float,
        object_image_scaling: List[int],
        tag_image_scaling: List[int],
        pole_height: float,
        padding: int=0
    ) -> None:
        """Init function.

        Args:
            cam_intr_path (str): path to the intrinsic calibration yml of the camera.
            cam_extr_path (str): path to the extrinsic calibration yml of the camera.
            image_shape (tuple): width and height of the images that this class operates on.
        """
        if dist is None:
            self.K = K
            self.intrinsic_calib_image_shape = image_shape
            self.image_shape = self.intrinsic_calib_image_shape
            self.undistorter = None
        else:
            self.undistorter = Undistorter("", K=K, dist=dist, image_shape=image_shape, new_width=new_width, new_height=new_height)
            self.intrinsic_calib_image_shape = self.undistorter.image_shape
            self.image_shape = self.intrinsic_calib_image_shape
            self.K = self.undistorter.K

        # np_extr = np.load(cam_extr_path, allow_pickle=True)
        self.R = rot_mat
        self.t = t_vec
        self.P = np.matmul(self.K, np.hstack([self.R, self.t.reshape((3, 1))]))
        self.Rt = np.hstack([self.R, np.reshape(self.t, (3, 1))])

        # print(f"self.P", self.P)

        # define a transformation matrix that transforms a pixel to a 3d coordinate with z=atr_height
        temp = np.zeros((4, 4))
        temp[0:2, :] = self.P[0:2, :]
        temp[3, :] = self.P[2, :]
        temp[2, 2] = 1
        temp[2, 3] = -atr_height
        self.atr_transformation_mat = np.linalg.inv(temp)

        # define a transformation matrix tat transforms a pixel to a 3d coordinate with z=0
        temp = np.zeros((4, 4))
        temp[0:2, :] = self.P[0:2, :]
        temp[3, :] = self.P[2, :]
        temp[2, 2] = 1
        temp[2, 3] = -pole_height
        self.pole_transformation_mat = np.linalg.inv(temp)

        temp = np.zeros((4, 4))
        temp[0:2, :] = self.P[0:2, :]
        temp[3, :] = self.P[2, :]
        temp[2, 2] = 1
        temp[2, 3] = 0.0
        self.floor_transformation_mat = np.linalg.inv(temp)

        self.object_image_scaling = object_image_scaling
        self.tag_image_scaling = tag_image_scaling

    def tag_detections_to_poses(
        self, detected_corners: np.ndarray, ids: np.ndarray, tag_type: str
    ) -> Tuple[
        List[Dict[str, list]], list
    ]:  # TODO instead of computing theta from only two points, we could use all four points,
        # alternatively using aruco pose estimation directly.
        """Transform tag detections to ATR Poses.

        Args:
            detected_corners (np.ndarray): np.ndarray of size (n_tags x 4 x 2) that specifies
                the pixel locations of each corner for n tag detections.
            ids (np.ndarray): n x 1 np.ndarray that specifies the ID of each of the
                n detected corners

        returns:
            tuple consisting of: a list containing a dict for each atr pose specifying center
            coordinate and theta, and a list of ids
        """
        if tag_type == "atr":
            transformation_mat = self.atr_transformation_mat
        elif tag_type == "pole":
            transformation_mat = self.pole_transformation_mat
        else:
            transformation_mat = np.zeros((1, 1))
            raise NameError("Invalid input tag_type to tag_detections_to_poses")

        detections_3d = []
        for _, corners in enumerate(detected_corners):
            detection_dict = {}
            corners = np.transpose(corners)
            corner0_3d = self.tag_pixels_to_3d_plane(corners[:, 0], transformation_mat)
            corner3_3d = self.tag_pixels_to_3d_plane(corners[:, 3], transformation_mat)
            x_vec = np.squeeze(np.subtract(corner0_3d, corner3_3d))
            angle_3d = np.arctan2(x_vec[1], x_vec[0])

            tag_center = np.mean(corners, axis=1)
            tag_center_3d = self.tag_pixels_to_3d_plane(tag_center, transformation_mat)

            detection_dict["tag_center_3d"] = tag_center_3d.squeeze().tolist()
            detection_dict["theta"] = angle_3d

            detections_3d.append(detection_dict)

        if ids.shape[0] == 1:
            ids_list = [ids.squeeze().tolist()]
        else:
            ids_list = ids.squeeze().tolist()

        return detections_3d, ids_list

    def tag_pixels_to_3d_plane(
        self, pixels: np.ndarray, transformation_mat: np.ndarray
    ) -> np.ndarray:
        """Transform pixels of object on floor level (z=0) to factory coordinate frame.

        By using a predefined height, the transformation can be pre-computed and solving the
        system of linear equations is reduced to a matrix multiplication.

        Args:
            pixel (np.ndarray): 2 x n matrix defining n pixels.

        returns:
            x_world (np.ndarray): 3 x n matrix defining n world coordinates.
        """
        if not isinstance(pixels, np.ndarray):
            raise NameError("Invalid input to tag_pixels_to_3d_plane")
        if pixels.ndim == 1:
            temp = np.zeros((4, 1))
            pixels = pixels.reshape((2, 1))
        else:
            temp = np.zeros((4, pixels.shape[1]))
        if self.tag_image_scaling:
            # TODO weird indexing. This is because the img size is written as e.g.,
            # [ 1512., 2688., 3. ] in the intrinsic calib yaml, while in the configuration files in factory_db it is written as e.g.,
            # image_shape_resized: [1024, 576]. Change the format in factory_db to be consistent with opencv.shape?
            if self.tag_image_scaling[0] > 0 and self.tag_image_scaling[1] > 0:
                pixels[0, :] = (
                    pixels[0, :]
                    * self.intrinsic_calib_image_shape[1]
                    / self.tag_image_scaling[0]
                )
                pixels[1, :] = (
                    pixels[1, :]
                    * self.intrinsic_calib_image_shape[0]
                    / self.tag_image_scaling[1]
                )
        temp[3, :] = 1
        temp[0:2, :] = pixels
        solution = np.matmul(transformation_mat, temp)
        x_world = np.divide(solution, solution[-1])

        return x_world[0:3, :]

    def floor_pixels_to_3d(self, pixels: np.ndarray) -> np.ndarray:
        """Transform pixels of object on floor level (z=0) to factory coordinate frame.

        By using a predefined height, the transformation can be pre-computed and solving the
        system of linear equations is reduced to a matrix multiplication.

        Args:
            pixel (np.ndarray): 2 x n matrix defining n pixels.

        returns:
            x_world (np.ndarray): 3 x n matrix defining n world coordinates.
        """
        if not isinstance(pixels, np.ndarray):
            raise NameError("Invalid input to floor_pixels_to_3d")
        else:
            assert pixels.ndim <= 2 and pixels.ndim >=1, f"pixels shape not accaptable: {pixels.shape}"
        if pixels.ndim == 1:
            temp = np.zeros((4, 1))
            pixels = pixels.reshape((2, 1))
        else:
            temp = np.zeros((4, pixels.shape[1]))
        # if self.object_image_scaling:
            # pixels[0, :] = (
            #     pixels[0, :]
            #     * self.intrinsic_calib_image_shape[1]
            #     / self.object_image_scaling[0]
            # )
            # pixels[1, :] = (
            #     pixels[1, :]
            #     * self.intrinsic_calib_image_shape[0]
            #     / self.object_image_scaling[1]
            # )
        temp[3, :] = 1
        temp[0:2, :] = pixels
        solution = np.matmul(self.floor_transformation_mat, temp)
        x_world = np.divide(solution, solution[-1])

        return x_world[0:3, :]
    
    def scale_pixels(self, pixels: np.ndarray, scale) -> np.ndarray:
        return scale*pixels

    def detected_objects_to_factory_floor(
        self, detected_objects: np.ndarray
    ) -> List[np.ndarray]:
        """Transform detected objects in image to 3D objects on factory floor (z=0 in FCF).

        Args:
            detected_corners (np.ndarray): np.ndarray of size (n_objects x m_points x 2) that specifies
                the pixel locations of each point in the polygon of length m that represent object n.

        returns:
            list of length n_objects where each entry is an np.ndarray of size (m_points x 3)
        """
        detections_3d = []
        for _, polygon_n in enumerate(detected_objects):
            detection_dict = {}
            polygon_n = np.transpose(polygon_n)
            if self.object_image_scaling is not None:
                polygon_n = self.scale_pixels(polygon_n, self.object_image_scaling)
            polygon_n_3d = self.floor_pixels_to_3d(polygon_n)

            detections_3d.append(polygon_n_3d.transpose())

        return detections_3d

    @staticmethod
    def world_coordinates_to_pixels(
        P: np.ndarray, world_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transform world coordinates to pixels:

        Args:
            P (np.ndarray): camera matrix used to do the world to pixel transformation.
            world_coordinates (np.ndarray): 3 x n np.ndarray defining n world coordinate points.

        returns:
            (np.ndarray): 2 x n array that defines pixel locations
        """
        temp = np.ones((4, world_coordinates.shape[1]))
        temp[0:3, :] = world_coordinates
        pixels = np.matmul(P, temp)
        pixels = np.divide(pixels, pixels[2, :])

        return pixels[0:2, :].astype(np.int32)

