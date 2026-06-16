


import cv2
import os

import numpy as np

from utils import *
import json
import time
import argparse

import numpy as np

from matplotlib.path import Path
from datetime import datetime



def view_existing_tracks(args):

    data_folder = "Data/confidential_tuve_dataset"
    trajectories_path = "JSON-data/final.json"
    cam_extr_dir = "volvo_calib/extr"
    cam_intr_dir = "volvo_calib/intr"



    config_path = os.path.join(data_folder, "dataset_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    available_cameras = config.keys()
    cameras = args.cameras.split(",")
    sub_folders = [str(x) for x in cameras if x in available_cameras]
    if len(sub_folders) == 0:
        raise ValueError(f"No valid cameras found in the config file. Available cameras: {available_cameras}")




    occupancy_map, cam_to_fac_dict = redis_test(args, cam_extr_dir, cam_intr_dir, sub_folders)


    # Opening JSON file
    f = open(trajectories_path)
    data = json.load(f)

    print("len(data)", len(data))

    if args.selected_sequences:
        time_windows = load_selected_time_windows()  # uses repo/data/selected_sequences.json by default
        for entry in time_windows:
            print(entry["start_time"], "->", entry["end_time"])


    time_window_index = 0
    view_interval = 1
    i = 0
    while i < len(data):
    # for i, d in enumerate(data):
        d = data[i]
        object_list = d["object_list"]
        time_stamp = d["time_stamp"]
        current_time = datetime.fromtimestamp(int(float(time_stamp)/1000)).strftime('%Y-%m-%d  %H:%M:%S')

        if args.selected_sequences:
            current_time_dt = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')

            entry = time_windows[time_window_index]
            if current_time_dt < entry["start_time"]:
                print(f"Current time {current_time_dt} is before the start of the time window {entry['start_time']}")
                i += 1
                continue
            elif current_time_dt > entry["end_time"]:
                time_window_index += 1
                if time_window_index >= len(time_windows):
                    print("No more time windows to display. Exiting.")
                    break
                print(f"Finished first sequence wtih end time {entry['end_time']}. Moving to next sequence starting at {time_windows[time_window_index]['start_time']}")
                continue


        bev_img_path = os.path.join(data_folder, "bev_img", str(time_stamp) + ".jpg")

        assert os.path.exists(bev_img_path), f"bev_img_path {bev_img_path} does not exist"
        bev_img = cv2.imread(bev_img_path)

        # if not object_list:
        #     continue


        for obj in object_list:
            # Support both legacy list format and new dictionary format.
            bbox_width_avg = None
            bbox_length_avg = None
            associated_polygon = None
            heading_rad = None
            if isinstance(obj, dict):
                id = obj["track_id"]
                x = obj["position_3d"]["x"]
                y = obj["position_3d"]["y"]
                vel_x = obj["velocity_3d"]["x"]
                vel_y = obj["velocity_3d"]["y"]
                bbox_moving_avg = obj.get("bbox_moving_avg", {})
                bbox_width_avg = bbox_moving_avg.get("width")
                bbox_length_avg = bbox_moving_avg.get("length")
                associated_polygon = obj.get("associated_polygon")
                heading_data = obj.get("heading", {})
                heading_rad = heading_data.get("rad")
            else:
                id = obj[0]
                x = obj[1]
                y = obj[2]
                vel_x = obj[3]
                vel_y = obj[4]
                if len(obj) >= 9:
                    bbox_width_avg = obj[7]
                    bbox_length_avg = obj[8]
            z = 0.
            pos = np.array([x, y, z])

            pos2 = np.array([x + vel_x, y + vel_y, z])


            traj_pix = occupancy_map.cam_fic.world_coordinates_to_pixels(occupancy_map.cam_fic.P, pos.reshape(3, 1))
            traj_pix2 = occupancy_map.cam_fic.world_coordinates_to_pixels(occupancy_map.cam_fic.P, pos2.reshape(3, 1))

            cv2.circle(bev_img, (int(traj_pix[0]), int(traj_pix[1])), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(
                bev_img,
                f"id: {id}",
                (int(traj_pix[0]), int(traj_pix[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            if associated_polygon is not None:
                polygon_arr = np.asarray(associated_polygon, dtype=np.float32)
                if polygon_arr.ndim == 2 and polygon_arr.shape[1] == 2 and polygon_arr.shape[0] >= 3:
                    polygon_cv = np.round(polygon_arr).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bev_img, [polygon_cv], isClosed=True, color=(255, 0, 255), thickness=2)
            if bbox_width_avg is not None and bbox_length_avg is not None:
                center = np.array([float(traj_pix[0]), float(traj_pix[1])], dtype=np.float32)
                if heading_rad is not None:
                    # length is along heading; width is orthogonal to heading
                    direction = np.array([np.cos(float(heading_rad)), np.sin(float(heading_rad))], dtype=np.float32)
                    orthogonal = np.array([-direction[1], direction[0]], dtype=np.float32)
                    half_l = float(bbox_length_avg) / 2.0
                    half_w = float(bbox_width_avg) / 2.0
                    corners = np.array([
                        center + half_l * direction + half_w * orthogonal,
                        center + half_l * direction - half_w * orthogonal,
                        center - half_l * direction - half_w * orthogonal,
                        center - half_l * direction + half_w * orthogonal,
                    ], dtype=np.float32)
                    corners_cv = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(bev_img, [corners_cv], isClosed=True, color=(0, 255, 255), thickness=2)
                else:
                    half_w = int(round(float(bbox_width_avg) / 2.0))
                    half_l = int(round(float(bbox_length_avg) / 2.0))
                    center_x, center_y = int(round(center[0])), int(round(center[1]))
                    cv2.rectangle(
                        bev_img,
                        (center_x - half_w, center_y - half_l),
                        (center_x + half_w, center_y + half_l),
                        (0, 255, 255),
                        2,
                    )
            if heading_rad is not None:
                heading_len = 25
                heading_end = (
                    int(round(float(traj_pix[0]) + heading_len * np.cos(float(heading_rad)))),
                    int(round(float(traj_pix[1]) + heading_len * np.sin(float(heading_rad)))),
                )
                cv2.arrowedLine(
                    bev_img,
                    (int(traj_pix[0]), int(traj_pix[1])),
                    heading_end,
                    (0, 255, 255),
                    thickness=2,
                )
            cv2.arrowedLine(bev_img, (int(traj_pix[0]), int(traj_pix[1])), (int(traj_pix2[0]), int(traj_pix2[1])), (0, 255, 0), thickness=3)


        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'FPS: {round(10 / view_interval, 2)}'
        font_scale = 0.5
        color = (255, 255, 255)  # White color
        thickness = 2

        # Display current_time just above FPS
        current_time_text = f'Time: {current_time}'
        current_time_size = cv2.getTextSize(current_time_text, font, font_scale, thickness)[0]
        current_time_x = bev_img.shape[1] - current_time_size[0] - 10
        current_time_y = bev_img.shape[0] - 10 - current_time_size[1] - 5
        cv2.putText(bev_img, current_time_text, (current_time_x, current_time_y), font, font_scale, color, thickness)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = bev_img.shape[1] - text_size[0] - 10
        text_y = bev_img.shape[0] - 10
        cv2.putText(bev_img, text, (text_x, text_y), font, font_scale, color, thickness)
        # Describe control commands
        control_text = [
            "Controls:",
            "q: Quit",
            "Right Arrow: Next frame",
            "Left Arrow: Previous frame",
            "Up Arrow: Increase FPS",
            "Down Arrow: Decrease FPS"
        ]

        # Print control commands in the bottom right quadrant of the image
        for idx, line in enumerate(control_text):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = bev_img.shape[1] * 3 // 4 - text_size[0] // 2
            text_y = bev_img.shape[0] * 3 // 4 + idx * (text_size[1] + 10)
            cv2.putText(bev_img, line, (text_x, text_y), font, font_scale, color, thickness)

        if args.selected_sequences:
            # Display time window in red text just above "Time: current time"
            time_window_entry = time_windows[time_window_index]
            window_text = f'Window {time_window_index}: {time_window_entry["start_time"].strftime("%H:%M:%S")} -> {time_window_entry["end_time"].strftime("%H:%M:%S")}'
            window_font_scale = 0.7
            window_color = (0, 0, 255)  # Red color
            window_thickness = 2
            window_text_size = cv2.getTextSize(window_text, font, window_font_scale, window_thickness)[0]
            window_text_x = bev_img.shape[1] - window_text_size[0] - 10
            window_text_y = text_y - window_text_size[1] - 10
            cv2.putText(bev_img, window_text, (window_text_x, window_text_y), font, window_font_scale, window_color, window_thickness)


        cv2.imshow("occupancy_map", bev_img)
        # Handle key inputs
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow key
            i += view_interval
        elif key == 81:  # Left arrow key
            i -= view_interval
        elif key == 82:  # Up arrow key
            view_interval -= 5
        elif key == 84:  # Down arrow key
            view_interval += 5
        view_interval = view_interval // 5 * 5
        view_interval = max(1, view_interval)



def create_image_view_matrix(img_list):
    """Create a matrix of images from a list of images"""
    # Get the height and width of the images
    height, width, _ = img_list[0].shape

    # Calculate the number of rows and columns for the image matrix
    num_images = len(img_list)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a blank matrix to store the images
    img_matrix = np.zeros((num_rows * height, num_cols * width, 3), dtype=np.uint8)

    # Loop through the images and add them to the matrix
    for idx, img in enumerate(img_list):
        row = idx // num_cols
        col = idx % num_cols
        img_matrix[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = img

    # Resize the image matrix to fit the screen
    screen_height, screen_width = 1080/1.5, 1920/1.5  # Example screen resolution
    scale = min(screen_width / img_matrix.shape[1], screen_height / img_matrix.shape[0])
    img_matrix = cv2.resize(img_matrix, (int(img_matrix.shape[1] * scale), int(img_matrix.shape[0] * scale)))

    return img_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    # parser.add_argument('--mode',
    #                     default='view',
    #                     const='view',
    #                     nargs='?',
    #                     choices=['view', 'track', 'write'],
    #                     help='select to view existing tracks, track live, or write new tracks (default: %(default)s)')
    # parser.add_argument(
    #     '--bev',
    #     action="store_true",
    #     help='bev flag')
    # parser.add_argument(
    #     '--obstacles',
    #     action="store_true",
    #     help='obstacles flag')
    parser.add_argument(
        '--selected_sequences',
        action="store_true",
        help='selected_sequences flag')
    # parser.add_argument(
    #     '--selected_sequence',
    #     type=int,
    #     default=None,
    #     help='selected sequence index')
    # parser.add_argument(
    #     '--view_interval',
    #     type=int,
    #     default=30,
    #     help='view interval (default: %(default)s)')
    parser.add_argument(
        '--contour_min_length',
        type=int,
        default=200,
        help='Kalman filter process noise scale.')

    parser.add_argument(
        '--cameras',
        type=str,
        default="160,161,162,163,164,165,166,167,169,170,171",
        help='Cameras.')
    args = parser.parse_args()
    return args
    


if __name__ == "__main__":
    args = parse_args()

    # print(args.mode)

    view_existing_tracks(args)

