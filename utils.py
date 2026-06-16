import os
import cv2
import os
import numpy as np
import json
from datetime import datetime
from cam_to_factory import CamToFactory
from oaf import OccupancyMap

def load_selected_time_windows(path=None, time_format="%Y-%m-%d %H:%M:%S"):
    """
    Load `data/selected_sequences.json` and parse `time_windows` into datetime objects.

    Returns
    -------
    list of dict
        Each dict contains keys `start_time` and `end_time` as `datetime` objects.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "data", "selected_sequences.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Selected sequences file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    time_windows = data.get("time_windows", [])
    parsed = []
    for entry in time_windows:
        s = entry.get("start_time")
        e = entry.get("end_time")
        if s is None or e is None:
            continue
        parsed.append({
            "start_time": datetime.strptime(s, time_format),
            "end_time": datetime.strptime(e, time_format),
        })
    return parsed



def redis_test(args, cam_extr_dir, cam_intr_dir, cam_ids):
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
        ] = 0.05
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
    DEBUG = False

    # cam_extr_dir = "volvo_calib/extr"
    # cam_intr_dir = "volvo_calib/intr"
    # cam_ids = "161,162,163,164,165,166,167,169,170,171"
    # cam_ids = cam_ids.split(",")

    config_parameters= redis_test_config()
    config_parameters["perception"]["overlapping_area_fusion"]["contour_min_length"] = args.contour_min_length

    cam_to_fac_dict = {}
    cam_roi_dict = {}
    print("cam_ids", cam_ids)
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
        print("img_shape", img_shape)
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


    return occupancy_map, cam_to_fac_dict

    # # choose directory
    # # dir_name = "images/stitching"
    # dir_name = img_dir
    # # img_list = [os.path.join(dir_name, f"cam{x}/image_1.png") for x in cam_ids]
    # img_list = [os.path.join(dir_name, f"cam{x}/image_1.jpg") for x in cam_ids]
    # for img in img_list:
    #     assert os.path.exists(img), f"{img} doesnt exist"
    # img_list = [cv2.imread(img) for img in img_list]
    
    # tot_img = None
    # tot_img_list = []
    # for cam_num, img in zip(cam_ids, img_list):
        
    #     img = cam_to_fac_dict[cam_num].undistorter.undistort(img)
    #     img_warped = occupancy_map.image_to_bev(img, cam_num)

    #     img_warped = img_warped.astype(np.float32)
    #     tot_img_list.append(img_warped)


    # tot_img = np.sum(tot_img_list, axis=0)
    # if np.max(tot_img) > 0:
    #     tot_img = tot_img / np.max(tot_img) * 255


    # img = np.array(tot_img.astype(np.uint8))

    # return img[:,:,::-1], occupancy_map

def create_bev_view(cam_ids, img_list, occupancy_map, cam_to_fac_dict):
    tot_img_list = []
    for cam_num, img in zip(cam_ids, img_list):
        
        img = cam_to_fac_dict[cam_num].undistorter.undistort(img)
        img_warped = occupancy_map.image_to_bev(img, cam_num)

        img_warped = img_warped.astype(np.float32)
        tot_img_list.append(img_warped)


    tot_img = np.sum(tot_img_list, axis=0)
    if np.max(tot_img) > 0:
        tot_img = tot_img / np.max(tot_img) * 255


    img = np.array(tot_img.astype(np.uint8))

    return img


def create_bev_res(cam_ids, img_list, occupancy_map, cam_to_fac_dict):
    tot_img_list = []
    for cam_num, img in zip(cam_ids, img_list):
        
        img = cam_to_fac_dict[cam_num].undistorter.undistort(img)
        img_warped = occupancy_map.image_to_bev(img, cam_num)

        img_warped = img_warped.astype(np.float32)
        tot_img_list.append(img_warped)


    tot_img = np.sum(tot_img_list, axis=0) / len(img_list)

    img = np.array(tot_img.astype(np.uint8))

    return img, tot_img_list

def update_bev_res(bev_res_prev, cam_ids, img_list, occupancy_map, cam_to_fac_dict):
    tot_img_list = []
    bev_indicator_list = []
    for cam_num, img in zip(cam_ids, img_list):
        
        img = cam_to_fac_dict[cam_num].undistorter.undistort(img)
        img_warped = occupancy_map.image_to_bev(img, cam_num)

        img_warped = img_warped.astype(np.float32)
        tot_img_list.append(img_warped)

        bev_indicator = occupancy_map.camera_view_indicator[cam_num].warped_view_mask
        bev_indicator_list.append(bev_indicator)

    # sum all per-view BEV results
    tot_img = np.sum(tot_img_list, axis=0)
    print("len(img_list)", len(img_list))

    assert np.allclose(tot_img[:,:,0], tot_img[:,:,1]) and np.allclose(tot_img[:,:,1], tot_img[:,:,2]), "tot_img channels are not equal"
    tot_img = tot_img[:,:,0]  # shape HxW
    tot_img = tot_img[..., None]  # shape HxWx1

    # average over the number of cameras that have view of that cell
    bev_indicator_list = np.array(bev_indicator_list)
    combined_view_indicator = np.sum(bev_indicator_list, axis=0)
    den = combined_view_indicator.astype(np.float32)  # shape HxW
    den_exp = np.where(den[..., None] > 0, den[..., None], 1.0)  # shape HxWx1, avoid divide-by-zero

    print(np.unique(den_exp))
    tot_img = tot_img.astype(np.float32)
    tot_img = tot_img / den_exp # weighted sum by the number of cameras with view of that cell
    tot_img = tot_img[:,:,0]

    # only update the cells in prev BEV that have view from current cameras
    update_mask = (den > 0).astype(np.float32)  # shape HxW
    tot_img = bev_res_prev * (1.0 - update_mask) + tot_img * update_mask

    img = np.array(tot_img.astype(np.uint8))

    print(np.unique(img))


    return img, combined_view_indicator

def update_bev_res_bevinput(bev_res_prev, cam_ids, bev_img_list, occupancy_map, score_thresh):
    tot_img_list = bev_img_list
    bev_indicator_list = []
    for cam_num, img in zip(cam_ids, bev_img_list):
        bev_indicator = occupancy_map.camera_view_indicator[cam_num].warped_view_mask
        bev_indicator_list.append(bev_indicator)

    # sum all per-view BEV results
    tot_img = np.sum(tot_img_list, axis=0)
    print("len(img_list)", len(bev_img_list))

    assert np.allclose(tot_img[:,:,0], tot_img[:,:,1]) and np.allclose(tot_img[:,:,1], tot_img[:,:,2]), "tot_img channels are not equal"
    tot_img = tot_img[:,:,0]  # shape HxW
    tot_img = tot_img[..., None]  # shape HxWx1

    # average over the number of cameras that have view of that cell
    bev_indicator_list = np.array(bev_indicator_list)
    combined_view_indicator = np.sum(bev_indicator_list, axis=0)
    den = combined_view_indicator.astype(np.float32)  # shape HxW
    den_exp = np.where(den[..., None] > 0, den[..., None], 1.0)  # shape HxWx1, avoid divide-by-zero

    print("np.unique(den_exp)", np.unique(den_exp))
    tot_img = tot_img.astype(np.float32)

    # Scale tot_img to the range 0 to 255
    print("min_val, max_val:", np.min(tot_img), np.max(tot_img))

    tot_img = tot_img / den_exp # weighted sum by the number of cameras with view of that cell
    tot_img = tot_img[:,:,0]
    print("min_val, max_val:", np.min(tot_img), np.max(tot_img))

    # only update the cells in prev BEV that have view from current cameras
    update_mask = (den > 0).astype(np.float32)  # shape HxW
    tot_img = bev_res_prev * (1.0 - update_mask) + tot_img * update_mask

    print("min_val, max_val:", np.min(tot_img), np.max(tot_img))
    tot_img = np.clip(tot_img, 0, 255)

    img = np.array(tot_img.astype(np.uint8))



    obstacle_map = img / 255 > score_thresh
    obstacle_map = obstacle_map.astype(np.uint8)

    return obstacle_map, img, combined_view_indicator


def forklift_load_and_fuse_bev_results(sub_folders, config, data_folder, score_threshold, data_index, augmented_data_folder=None):

        bev_res_per_cam = []
        for sub_folder in sub_folders:
            img_name = config[sub_folder][data_index]

            if img_name is None:
                bev_res_cam = np.zeros((470, 1528, 3), dtype=np.uint8)
            else:
                bev_res_path = os.path.join(data_folder, "bev_res_per_cam", sub_folder, img_name)
                if not os.path.exists(bev_res_path):
                    raise Exception(f"bev_res_path {bev_res_path} does not exist")
                if augmented_data_folder is not None:
                    bev_res_path_aug = os.path.join(augmented_data_folder, "bev_res_per_cam", sub_folder, img_name)
                    if os.path.exists(bev_res_path_aug):
                        bev_res_path = bev_res_path_aug
                        
                bev_res_cam = cv2.imread(bev_res_path)
            bev_res_cam = bev_res_cam[:,:,0]
            bev_res_per_cam.append(bev_res_cam)

        bev_res = np.array(bev_res_per_cam)
        max_bev = np.max(bev_res, axis=0)
        bev_res_mean = np.repeat(max_bev[:, :, None], 3, axis=2)
        bev_res = (max_bev / 255.0) > score_threshold
        bev_res = bev_res.astype(np.uint8)

        return bev_res, bev_res_mean

def load_obstacle_bev_results(sub_folders, config, data_folder, data_index, augmented_data_folder=None):
    cam_list = []
    bev_res_per_cam_obstacles = []
    for sub_folder in sub_folders:
        img_name = config[sub_folder][data_index]

        if img_name is not None:
            bev_res_path = os.path.join(data_folder, "bev_res_per_cam_obstacles", sub_folder, img_name)
            if not os.path.exists(bev_res_path):
                raise Exception(f"bev_res_path {bev_res_path} does not exist")
            if augmented_data_folder is not None:
                bev_res_path_aug = os.path.join(augmented_data_folder, "bev_res_per_cam_obstacles", sub_folder, img_name)
                if os.path.exists(bev_res_path_aug):
                    bev_res_path = bev_res_path_aug

            bev_res_cam = cv2.imread(bev_res_path)
            bev_res_per_cam_obstacles.append(bev_res_cam)
            cam_list.append(sub_folder)

    return cam_list, bev_res_per_cam_obstacles