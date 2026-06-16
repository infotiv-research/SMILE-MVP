"""This module provides functionality for keeping an up-to-date occupancy map.

The input to this module is the predicted obstacles from different cameras (independent predictions)
and the location of the ATRs in factory coordinate frame. Given such predictions at different time stamps,
this module maintains an occupancy map that represents the currently best
prediction of obstacle locations considering both the current and past predictions of all cameras.
"""

import cv2
import os
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from queue import Queue
import argparse

from cam_to_factory import CamToFactory


class CameraViewIndicator:
    """Class that holds the information about the 'view' of a camera.
    
    The view (ROI) is set manually through camera_roi_path or computed and set to a default view (considering the camera calibration).
    Additionally, functions are provided to be able to update the view (ROI) based on the ATRs' positions. 
     """
    def __init__(self, camera: CamToFactory, camera_fic: CamToFactory,
        occupancy_map_shape, atr_height, atr_width, atr_length, robot_padding, roi_coordinates: Optional[np.ndarray] = None) -> None:
        """Create the update indicator for a certain camera.

        This indicator states what region of the factory occupancy map the camera is responsible
        for updating. If there exists a camera_roi file for the specified camera, then the roi-corners
        (which states the 3d corners of the rectangular area on the factory floor that is visible
        from the camera) will be loaded from that file. Otherwise, the roi-corners are computed explicitly by
        considering the camera intrinsic and extrinsic calibration.

        Args:
            camera_id (str): id of the camera.

        """
        self.camera = camera
        self.camera_fic = camera_fic
        self.occupancy_map_shape = occupancy_map_shape
        self.atr_height = atr_height
        self.atr_width = atr_width
        self.atr_length = atr_length
        self.robot_padding = robot_padding
        # load roi_coordinates through the provided camera_roi_dict
        # roi_coordinates = None
        # if not camera_roi_path is None:
        #     if os.path.exists(camera_roi_path):
        #         with open(camera_roi_path) as f:
        #             data = yaml.load(f, Loader=SafeLoader)
        #             roi_coordinates = np.array(data["roi_world_coordinates"])
        
        # if roi_coordinates have not been specified in camera_roi_dict, compute the default roi_coordinates
        # if roi_coordinates is None:
        image_shape = camera.image_shape
        print("image shape: ", image_shape)
        # image_shape = [x[0] for x in image_shape]
        image_shape = [image_shape[1], image_shape[0]]
        # print("image shape: ", image_shape)
        # print("image_shape ########################: ", image_shape)
        # print("image_shape: ", image_shape[1])
        roi_corner1 = camera.floor_pixels_to_3d(
            np.array([image_shape[0] - 1, image_shape[1] - 1])
        )
        floor_pixel1 = [image_shape[0] - 1, image_shape[1] - 1]

        roi_corner2 = camera.floor_pixels_to_3d(np.array([0, 0]))
        floor_pixel2 = [0, 0]

        roi_corner3 = camera.floor_pixels_to_3d(np.array([0, image_shape[1] - 1]))
        floor_pixel3 = [0, image_shape[1] - 1]

        roi_corner4 = camera.floor_pixels_to_3d(np.array([image_shape[0] - 1, 0]))
        floor_pixel4 = [image_shape[0] - 1, 0]


        floor_pixels = [floor_pixel1, floor_pixel3, floor_pixel2, floor_pixel4]
        roi_corners = [roi_corner1, roi_corner3, roi_corner2, roi_corner4]
        # else:
        #     roi_corners = [
        #         roi_coordinates[:, i].reshape(3, 1)
        #         for i in range(roi_coordinates.shape[1])
        #     ]
        
        print("roi_corners", roi_corners)

        print("min(roi_corners)", np.min(np.array(roi_corners)[:,0]))
        print("max(roi_corners)", np.max(np.array(roi_corners)[:,0]))
        print("min(roi_corners)", np.min(np.array(roi_corners)[:,1]))
        print("max(roi_corners)", np.max(np.array(roi_corners)[:,1]))

        occupancy_map_roi = []
        for corner in roi_corners:
            occupancy_map_roi.append(
                CamToFactory.world_coordinates_to_pixels(camera_fic.P, corner)
            )

        occupancy_map_roi = np.hstack(occupancy_map_roi)

        temp1 = occupancy_map_roi.transpose().astype(np.float32)
        temp2 = np.array(floor_pixels).astype(np.float32)
        print("temp1:", temp1)
        print("temp2:", temp2)
        self.warp_mat = cv2.getPerspectiveTransform(temp2, temp1[:,:])

        # create an image of ones in the camera image space and warp it to occupancy-map space
        ones_img = np.ones((int(image_shape[1]), int(image_shape[0])), dtype=np.uint8)
        warped_ones = cv2.warpPerspective(ones_img, self.warp_mat, (int(occupancy_map_shape[1]), int(occupancy_map_shape[0])))
        # binary mask of the warped area (1 where camera sees the occupancy map)
        self.warped_view_mask = (warped_ones > 0).astype(np.uint8)


        # x_min = np.min(occupancy_map_roi[1, :])
        # x_max = np.max(occupancy_map_roi[1, :])
        # y_min = np.min(occupancy_map_roi[0, :])
        # y_max = np.max(occupancy_map_roi[0, :])


        print("occupancy_map_shape: ", occupancy_map_shape)

        x_min = np.min(occupancy_map_roi[0, :])
        x_max = np.max(occupancy_map_roi[0, :])
        y_min = np.min(occupancy_map_roi[1, :])
        y_max = np.max(occupancy_map_roi[1, :])

        print("x_min", x_min)
        print("x_max", x_max)
        print("y_min", y_min)
        print("y_max", y_max)

        assert x_min > 0 and y_min > 0, "camera view indicator mustn't be outside of the initialized occupancy map. Map must be made bigger, or view indicator must be restricted"
        assert x_max < occupancy_map_shape[1] and y_max < occupancy_map_shape[0], "camera view indicator mustn't be outside of the initialized occupancy map. Map must be made bigger, or view indicator must be restricted"

        roi = (x_min, x_max, y_min, y_max)
        x_range = [x_min, x_max]
        y_range = [y_min, y_max]

        occupancy_map_roi[0, :] -= y_min
        occupancy_map_roi[1, :] -= x_min
        # occupancy_map_roi[0, :] -= x_min
        # occupancy_map_roi[1, :] -= y_min

        print("occupancy_map_roi: ", occupancy_map_roi)

        occupancy_map_roi = np.transpose(occupancy_map_roi)
        occupancy_map_roi = np.reshape(
            occupancy_map_roi, (occupancy_map_roi.shape[0], 1, 2)
        )



        update_indicator = np.zeros(
            (int(roi[1] - roi[0]), int(roi[3] - roi[2]))
        ).astype(np.uint8)
        print("update_indicator.shape", update_indicator.shape)
        cv2.fillPoly(update_indicator, [occupancy_map_roi], (1, 1, 1))
        update_indicator = update_indicator.transpose()

        self.x_range = x_range
        self.y_range = y_range
        self.roi = roi
        print("roi", roi)

        # when initialized, the "current" indicator is the same as the default indicator
        self.indicator_default = update_indicator
        self.indicator = update_indicator


class OccupancyMap:
    """Class that handles merging obstacles from different cameras by maintaining an occupancy map/grid."""

    def __init__(
        self,
        config_parameters: Dict,
        camera_dict: Dict[str, CamToFactory],
        camera_roi_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        """Init function.

        Args:
            config_file (str): path to the config file for this class.
        """
        self.top_left_corner, self.bottom_right_corner = self.find_min_max_coordinates(camera_dict, camera_roi_dict)

        print("########################################################")
        print(self.top_left_corner)
        print(self.bottom_right_corner)
        print("########################################################")

        # initialize general parameters
        self.save_index = 0
        self.camera_roi_dict = camera_roi_dict
        # self.toy_indicator: Optional[np.ndarray] = None

        # define and load config parameters
        self.contour_min_length: int
        self.atr_height: float
        self.atr_width: float
        self.atr_length: float
        self.robot_padding: float
        self.occupancy_grid_resolution: float
        self.object_padding: float
        # self.top_left_corner: np.ndarray
        # self.bottom_right_corner: np.ndarray
        self.object_low_pass_value: int
        self.adjacency_dict: Dict[str, List[str]]
        self.load_config_params(config_parameters)

        # create the "fictional" camera that defines the image-to-world correspondence for the occupancy map
        self.cam_fic = self.create_fictional_camera(
            self.top_left_corner, self.bottom_right_corner, self.occupancy_grid_resolution
        )

        # define the main occupancy map
        self.occupancy_map = np.zeros(
            (self.cam_fic.image_shape[1], self.cam_fic.image_shape[0])
        )

        # define a Queue that holds previous occupancy map (i.e., the history of occupancy maps through time)
        if not self.object_low_pass_value == 0:
            self.occupancy_map_history = Queue(maxsize=self.object_low_pass_value)
        else:
            self.occupancy_map_history = None

        # define an occupancy map that hold obstacle that have been consistent over time
        self.occ_map_temp_consistent = np.zeros(
            (self.cam_fic.image_shape[1], self.cam_fic.image_shape[0])
        )

        self.camera_dict = camera_dict
        camera_ids = list(camera_dict.keys())
        self.camera_view_indicator: Dict[str, CameraViewIndicator] = {}

        # for i, cam_id in enumerate(camera_ids):
        #     K, image_shape, P, R, t = Camera.load_camera_params_from_files(cam_intr_path_list[i], cam_extr_path_list[i])
        #     self.camera_dict[cam_id] = Camera(K, (image_shape[1], image_shape[0]), P, R, t)

            # for each camera, create the default update indicator. Since this indicator is defined in the "fictional camera frame" we need to feed the 
            # fictional camera matrix P as input
        for cam_id, camtofact in self.camera_dict.items():
            if not self.camera_roi_dict is None:
                self.camera_view_indicator[cam_id] = CameraViewIndicator(camtofact, self.cam_fic,
                    self.occupancy_map.shape, self.atr_height, self.atr_width, self.atr_length, self.robot_padding, self.camera_roi_dict[cam_id])
            else:
                self.camera_view_indicator[cam_id] = CameraViewIndicator(camtofact, self.cam_fic,
                    self.occupancy_map.shape, self.atr_height, self.atr_width, self.atr_length, self.robot_padding)

        # define a dictionary that hold an occupancy map per camera
        self.occupancy_map_per_camera = {}
        for _, cam_id in enumerate(camera_ids):
            self.occupancy_map_per_camera[cam_id] = np.zeros(
                (
                    self.camera_view_indicator[cam_id].x_range[1]
                    - self.camera_view_indicator[cam_id].x_range[0],
                    self.camera_view_indicator[cam_id].y_range[1]
                    - self.camera_view_indicator[cam_id].y_range[0],
                )
            ).astype(np.uint8)

        # define variables that stores the "merged" obstacles, in 2D and 3D respectively
        # self.object_list: Optional[List] = []
        # self.object_list_3d: Optional[List] = None

        self.find_overlap()

    def load_config_params(self, config_parameters):
        self.contour_min_length = config_parameters["perception"][
            "overlapping_area_fusion"
        ]["contour_min_length"]
        self.atr_height = config_parameters["robots"]["geometry"]["height"]
        self.atr_width = config_parameters["robots"]["geometry"]["width"]
        self.atr_length = config_parameters["robots"]["geometry"]["length"]
        self.robot_padding = config_parameters["perception"]["overlapping_area_fusion"][
            "robot_padding"
        ]
        self.occupancy_grid_resolution = config_parameters["perception"][
            "overlapping_area_fusion"
        ]["occupancy_grid_resolution"]
        self.object_padding = config_parameters["perception"][
            "overlapping_area_fusion"
        ]["object_padding"]
        # self.top_left_corner = np.array(
        #     config_parameters["perception"]["overlapping_area_fusion"][
        #         "top_left_corner"
        #     ]
        # )
        # self.bottom_right_corner = np.array(
        #     config_parameters["perception"]["overlapping_area_fusion"][
        #         "bottom_right_corner"
        #     ]
        # )
        self.object_low_pass_value = config_parameters["perception"][
            "overlapping_area_fusion"
        ]["object_low_pass_value"]

        self.adjacency_dict = config_parameters["perception"][
            "overlapping_area_fusion"
        ]["adjacency_dict"]

    def create_fictional_camera(
        self, point1_3d: np.ndarray, point2_3d: np.ndarray, resolution: float
    ) -> CamToFactory:
        """Create a camera matrix for a fictional camera of which the pixels in the image correspond
        to a regularly spaced square grid in factory coordinate frame with a distance of "resolution" meters
        between each grid point.

        The grid to which the camera matrix corresponds will be alligned with the factory coordinate frame.
        The input to the function is the 3D coordinates that the top left corner and bottom right corner of the image
        should correspond to. The size of the image is calculated based on the resolution such that the distance
        between the grid point to which the image pixels correspond will be exactly equal to resolution meters.
        Note that the system of linear equations solved herein is given by K[R t]X = lambda*x.
        Since the rotation is orthogonal to the factory floor, lambda will be the same for both points.
        Furthermore, t can be fixed arbitrarily, since there will exist a solution regardless of the value of t.
        This gives 4 + 1 unkown variables, while there are 6 equations given two point correspondences.
        Therefore, the 6th equation can be removed.

        Args:
            point1_3d (np.ndarray): homogeneous 3d point to which the top left corner of the image correspond
            point2_3d (np.ndarray): homogeneous 3d point to which the bottom right corner of the image correspond
            resolution (float): the desired distance between the occupancy grid points to which the image pixels
                correspond. The distance is measured in meters.

        returns:
            Camera matrix P, K and image width and height.
        """
        # round the input 3D point to 10 cm intervals
        point1_3d = np.ceil(point1_3d * 10) / 10
        point2_3d = np.ceil(point2_3d * 10) / 10

        # print("point1_3d", point1_3d)
        # print("point2_3d", point2_3d)

        x_view = np.abs(point1_3d[0] - point2_3d[0])
        y_view = np.abs(point1_3d[1] - point2_3d[1])

        image_width = int(x_view / resolution)
        image_height = int(y_view / resolution)

        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        # R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        t = np.array([30.0, 0.0, 4.0])

        Rt = np.hstack([R, np.reshape(t, (3, 1))])

        u1 = np.array([0.0, 0.0, 1.0])
        u2 = np.array([image_width, image_height])
        a = np.concatenate([u1, u2])

        Xc1 = np.matmul(Rt, point1_3d)
        Xc2 = np.matmul(Rt, point2_3d)

        M = np.zeros((5, 5))

        M[0, 0] = Xc1[0]
        M[0, 2] = Xc1[2]
        M[1, 1] = Xc1[1]
        M[1, 3] = Xc1[2]
        M[2, 4] = Xc1[2]

        point_iter = 1
        M[3 * point_iter + 0, 0] = Xc2[0]
        M[3 * point_iter + 0, 2] = Xc2[2]
        M[3 * point_iter + 1, 1] = Xc2[1]
        M[3 * point_iter + 1, 3] = Xc2[2]
        # M[3*point_iter + 2, 4] = Xc2[2]

        solution = np.matmul(np.linalg.inv(M), a)
        solution = np.divide(solution, solution[-1])

        K = np.array(
            [[solution[0], 0, solution[2]], [0, solution[1], solution[3]], [0, 0, 1]]
        )

        P = np.matmul(K, Rt)

        return CamToFactory(K=K, dist=None,
                             image_shape=(image_width, image_height),
                             rot_mat=R, 
                             t_vec=t,
                             new_width=None,
                             new_height=None,
                             atr_height=0.,
                             object_image_scaling=[],
                             tag_image_scaling=[],
                             pole_height=0.)
    
    def find_overlap(self):
        adjacency_dict = {}
        for cam_id, _ in self.camera_dict.items():
            adjacency_dict[cam_id] = []
            indicator = self.camera_view_indicator[cam_id].indicator_default
            roi = self.camera_view_indicator[cam_id].roi

            print("find overlap, indicator.shape", indicator.shape)
            print("find overlap, roi", roi)

            indicator_full = self.transform_small_to_full_scale(
                indicator, roi
            ).astype(np.bool_)

            for cam_id2, camtofact in self.camera_dict.items():
                if cam_id2 == cam_id:
                    continue
                else:
                    indicator = self.camera_view_indicator[cam_id2].indicator_default
                    roi = self.camera_view_indicator[cam_id2].roi
                    indicator_full2 = self.transform_small_to_full_scale(
                        indicator, roi
                    ).astype(np.bool_)
                    if np.sum(np.logical_and(indicator_full, indicator_full2)) > 0:
                        adjacency_dict[cam_id].append(cam_id2)

        print("################################################")
        print("adjacency_dict", adjacency_dict)

        for cam, adj_list in adjacency_dict.items():
            for adj in adj_list:
                assert cam in adjacency_dict[adj], f"Adjacency dict is incorrect since {adj} is adjacent to {cam} but {cam} is not adjacent to {adj}"


        self.adjacency_dict = adjacency_dict

    def find_min_max_coordinates(self, camtofact_dict, cam_roi_dict):
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for camtofact, cam_roi in zip(camtofact_dict.values(), cam_roi_dict.values()):
            print("ca_roi: ", cam_roi)
            for p in cam_roi.transpose():
                assert p.size == 3 and np.allclose(p[-1], np.array(0.)), f"cam_roi should list points in 3D space [x, y, 0.] but got: {p}"
                if p[0] < x_min:
                    x_min = p[0]
                if p[0] > x_max:
                    x_max = p[0]
                if p[1] < y_min:
                    y_min = p[1]
                if p[1] > y_max:
                    y_max = p[1]
        top_left_corner = np.array([x_min - 1.0, y_max + 1.0, 0., 1.])
        bottom_right_corner = np.array([x_max + 1.0, y_min - 1.0, 0., 1.])
        return top_left_corner, bottom_right_corner

    # specific occupancy map stuff
    def get_view_area(self, camera_ids: List[str]) -> Tuple[int, int, int, int]:
        """Get the entire view area (rectangular) of any number of cameras.

        Args:
            camera_ids (list): list of camera ids

        returns:
            x_min, x_max, y_min, y_max describing the view area.
        """
        x_range_list = []
        y_range_list = []

        for cam_id in camera_ids:
            x_range = self.camera_view_indicator[cam_id].x_range
            y_range = self.camera_view_indicator[cam_id].y_range
            x_range_list.append(x_range[0])
            x_range_list.append(x_range[1])
            y_range_list.append(y_range[0])
            y_range_list.append(y_range[1])

        x_min = int(np.min(x_range_list))
        x_max = int(np.max(x_range_list))
        y_min = int(np.min(y_range_list))
        y_max = int(np.max(y_range_list))

        return (x_min, x_max, y_min, y_max)

    # oaf function taht operates on asingle occ map
    def transform_occupancy_map(
        self,
        occupancy: np.ndarray,
        source_roi: Tuple[int, int, int, int],
        dest_roi: Tuple[int, int, int, int],
    ):
        """Transform occupancy map of a certain camera into a desired view."""
        (x_min, x_max, y_min, y_max) = dest_roi
        resized_occupancy = np.zeros((x_max - x_min, y_max - y_min), dtype=np.uint8)

        x_range = [source_roi[0], source_roi[1]]
        y_range = [source_roi[2], source_roi[3]]

        # x_range = self.camera_view_indicator[camera_id]["x_range"]
        # y_range = self.camera_view_indicator[camera_id]["y_range"]

        # TODO raise exception if e.g. x_range - x_min < 0
        resized_occupancy[
            max(0, x_range[0] - x_min) : x_range[1] - x_min,
            max(0, y_range[0] - y_min) : y_range[1] - y_min,
        ] = occupancy[
            max(0, x_min - x_range[0]) : x_max - x_range[0],
            max(0, y_min - y_range[0]) : y_max - y_range[0],
        ]

        return resized_occupancy

    # oaf function that operates on a single occ map
    def transform_small_to_full_scale(self, occupancy, roi):
        """Transform a small occupancy map derived from a camera to the full scale occupancy map."""
        temp = np.zeros(self.occupancy_map.shape)
        (x_min, x_max, y_min, y_max) = roi
        # temp[x_min:x_max, y_min:y_max] = occupancy
        temp[y_min:y_max, x_min:x_max] = occupancy

        return temp
    
    def image_to_bev(self, img, cam_id):
        
        # pixels = CamToFactory.world_coordinates_to_pixels(self.camera_dict[cam_id].P, self.occupancy_map)

        img_warped = cv2.warpPerspective(img, self.camera_view_indicator[cam_id].warp_mat, (self.occupancy_map.shape[1], self.occupancy_map.shape[0]))

        return img_warped

    def filter_occupancy_map(self) -> np.ndarray:
        """Apply morphological operations to filter the occupancy map."""

        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.occupancy_map = cv2.erode(self.occupancy_map, kernel, iterations=1)
        self.occupancy_map = cv2.dilate(self.occupancy_map, kernel, iterations=1)

        return self.occupancy_map

    @staticmethod
    def find_objects_in_occupancy_map(
        occupancy_map: np.ndarray, contour_min_length
    ) -> List[np.ndarray]:
        """Find contours in the current occupancy map and create convex hulls from these."""
        # find the contours in the occupancy map
        contours, _ = cv2.findContours(
            occupancy_map.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Find the convex hull object for each contour of significant length
        hull_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > contour_min_length:
            # hull = cv2.convexHull(contour) # TODO add back this? Currently, the objects are not convex
                hull = contour
                hull_list.append(hull)

        return hull_list

    def extract_3d_objects_from_map(self):
        object_list_3d = None

        start_temporal = time.time_ns()/1e6
        object_list = self.find_objects_in_occupancy_map(
            self.occupancy_map, self.contour_min_length
        )
        object_list_3d = self.object_list_to_3d(object_list)

        return object_list_3d

    def find_bounding_boxes(self):
        boxes = None

        start_temporal = time.time_ns()/1e6
        object_list = self.find_objects_in_occupancy_map(
            self.occupancy_map, self.contour_min_length
        )
        # object_list_3d = self.object_list_to_3d(object_list)
        boxes = np.zeros((len(object_list), 4))
        for i, ob in enumerate(object_list):
            ob = ob.squeeze()
            x = ob[:, 0]
            y = ob[:, 1]
            (bb_left, bb_top, bb_width, bb_height) = (np.min(x), np.min(y),
                                                    np.max(x)-np.min(x), np.max(y) - np.min(y))
            boxes[i, :] = [bb_left, bb_top, bb_width, bb_height]

        return boxes, object_list
    



def redis_test(args):
    def redis_test_config():
        """Create a configuration to be able to run the unit test."""
        config_parameters = {}
        config_parameters["perception"] = {}
        config_parameters["perception"]["overlapping_area_fusion"] = {}
        config_parameters["perception"]["overlapping_area_fusion"][
            "contour_min_length"
        ] = 30
        config_parameters["perception"]["overlapping_area_fusion"]["robot_padding"] = 0.20
        config_parameters["perception"]["overlapping_area_fusion"]["object_padding"] = 0
        config_parameters["perception"]["overlapping_area_fusion"]["object_padding_in_image"] = 0
        config_parameters["perception"]["overlapping_area_fusion"][
            "occupancy_grid_resolution"
        ] = 0.01
        config_parameters["perception"]["overlapping_area_fusion"][
            "top_left_corner"
        # ] = [-200.0, 500.0, 0.0, 1.0]
        ] = [-25.0, 20.0, 0.0, 1.0]
        config_parameters["perception"]["overlapping_area_fusion"][
            "bottom_right_corner"
        # ] = [400.0, -100.0, 0.0, 1.0]
        ] = [45.0, -15.0, 0.0, 1.0]
        config_parameters["perception"]["overlapping_area_fusion"][
            "object_low_pass_value"
        ] = 0
        config_parameters["perception"]["overlapping_area_fusion"]["adjacency_dict"] = {
            "140": [],#["141"],
            "141": [],#["142"],
            "142": [],#["143"],
            "143": [],#["144"],
            "144": [],#["145"],
            "145": [],#["146"],
            "146": [],#["147"],
            "147": [],#["148"],
            "148": [],#["147"], 
            "149": [],#["147"], 
            "150": [],#["147"], 
            "151": [],#["147"], 
            "152": [],#["147"], 
        }
        config_parameters["perception"]["overlapping_area_fusion"][
            "atr_save_duration"
        ] = 5.0

        config_parameters["robots"] = {}
        config_parameters["robots"]["geometry"] = {}
        config_parameters["robots"]["geometry"]["height"] = 0.74
        config_parameters["robots"]["geometry"]["length"] = 0.8
        config_parameters["robots"]["geometry"]["width"] = 0.5


        return config_parameters

    def load_camera_params(intrinsic=None, extrinsic=None, roi=None):
        mtx = None
        dist = None
        image_shape = None
        rot_mat = None
        t_vec = None
        # camera_matrix = None
        if intrinsic is not None:
            cv_file = cv2.FileStorage(intrinsic, cv2.FILE_STORAGE_READ)
            mtx = cv_file.getNode("K").mat()
            dist = cv_file.getNode("D").mat()
            image_shape = cv_file.getNode("image_shape").mat()
        if extrinsic is not None:
            cv_file = cv2.FileStorage(extrinsic, cv2.FILE_STORAGE_READ)
            rot_mat = cv_file.getNode("rot_mat").mat()
            t_vec = cv_file.getNode("t_vec").mat()
        roi_world_coordinates = None
        if roi is not None:
            cv_file = cv2.FileStorage(roi, cv2.FILE_STORAGE_READ)
            roi_world_coordinates = cv_file.getNode("roi_world_coordinates").mat()
        return (mtx, dist ,image_shape), (rot_mat, t_vec), roi_world_coordinates

    global DEBUG
    DEBUG = args.debug

    cam_extr_dir = "volvo_calib/extr"
    cam_intr_dir = "volvo_calib/intr"
    cam_ids = args.cam_ids
    cam_ids = cam_ids.split(",")

    config_parameters= redis_test_config()

    cam_to_fac_dict = {}
    cam_roi_dict = {}
    for cam_id in cam_ids:
        # load camera parameters
        intrinsic = os.path.join(cam_intr_dir, f"{cam_id}_640x360.yaml")
        # intrinsic = os.path.join(cam_intr_dir, f"{cam_id}.yaml")
        extrinsic = os.path.join(cam_extr_dir, f"{cam_id}.yaml")
        assert os.path.exists(intrinsic), f"path doesnt exist: {intrinsic}"
        assert os.path.exists(extrinsic), f"path doesnt exist: {extrinsic}"
        intr, extr, _ = load_camera_params(intrinsic, extrinsic)

        # create coordinate transformation object
        cam_to_fact = CamToFactory(*intr, *extr, new_width=None, new_height=None,
                                   atr_height=0.74, object_image_scaling=None,
                                   tag_image_scaling=[],
                                   pole_height=0.915, padding=config_parameters["perception"]["overlapping_area_fusion"]["object_padding_in_image"])
        cam_to_fac_dict[cam_id] = cam_to_fact

        img_shape = [x[0] for x in intr[-1]]
        img_shape = [img_shape[1], img_shape[0]]
        floor_pixels = []
        floor_pixels.append([img_shape[0] - 1, img_shape[1] - 1])
        floor_pixels.append([0, 0])
        floor_pixels.append([0, img_shape[1] - 1])
        floor_pixels.append([img_shape[0] - 1, 0])
        floor_pixels = np.array(floor_pixels)

        roi = cam_to_fact.floor_pixels_to_3d(floor_pixels.transpose())
        cam_roi_dict[cam_id] = roi


    occupancy_map = OccupancyMap(
        config_parameters,
        cam_to_fac_dict,
        camera_roi_dict=cam_roi_dict,
    )


    # choose directory
    dir_name = "images/stitching"
    # img_list = [os.path.join(dir_name, f"cam{x}/image_1.png") for x in cam_ids]
    img_list = [os.path.join(dir_name, f"cam{x}/image_1.jpg") for x in cam_ids]
    for img in img_list:
        assert os.path.exists(img), f"{img} doesnt exist"
    img_list = [cv2.imread(img) for img in img_list]
    

    # choose if images should be rotated and save them in a new directory in that case
    # dst_dir_name = "images/stitching3"
    # img_list = [cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE) for x in img_list]
    # for i , img in enumerate(img_list):
    #     if not os.path.exists(os.path.join(dst_dir_name, f"cam{cam_ids[i]}/")):
    #         os.makedirs(os.path.join(dst_dir_name, f"cam{cam_ids[i]}/"))
    #     print("writing image to ", os.path.join(dst_dir_name, f"cam{cam_ids[i]}/image_1.png"))
    #     cv2.imwrite(os.path.join(dst_dir_name, f"cam{cam_ids[i]}/image_1.png"), img)


    # img_list = [cv2.resize(x, (640,360)) for x in img_list]

    

    tot_img = None
    tot_img_list = []
    for cam_num, img in zip(cam_ids, img_list):
        
        img = cam_to_fac_dict[cam_num].undistorter.undistort(img)
        img_warped = occupancy_map.image_to_bev(img, cam_num)

        img_warped = img_warped.astype(np.float32)
        tot_img_list.append(img_warped)


    tot_img = np.sum(tot_img_list, axis=0)
    if np.max(tot_img) > 0:
        tot_img = tot_img / np.max(tot_img) * 255

    cv2.namedWindow("Occupancy_map", cv2.WINDOW_NORMAL)
    cv2.imshow("Occupancy_map", np.array(tot_img.astype(np.uint8)))
    cv2.imwrite(os.path.join(dir_name, "stitched.jpg"), np.array(tot_img.astype(np.uint8)))
    cv2.waitKey(0)



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument(
        '--cam_ids',
        type=str,
        default="140,141,142,143,144,145,146,147,148,149,150,151,152",
        help='Specify cam ids in the form of a string, e.g.: 149,150,151')
    parser.add_argument(
        '--debug',
        action="store_true",
        help='debug flag')
    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    # python oaf.py --cam_ids "161,162,163,164,165,166,167,169,170,171
    # adding camera 168 makes the algorithm slow since it has such a "low" angle, but it should also work.
    args = parse_args()
    DEBUG = args.debug
    redis_test(args)