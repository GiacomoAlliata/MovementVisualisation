import cv2
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import mediapipe as mp

from process.get_features import compute_pose_and_mask, compute_depth, compute_mask


kernel_size = (5, 5)
FIXED_WIDTH = 360

def init_video_writer(video_cap, output_fp):
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))
    frame_size = (frame_width,frame_height)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer objects
    video_writer = cv2.VideoWriter(output_fp, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    return video_writer


def inpaint_video(video_path, output_fp = "results/test.mp4", 
                  every_n = 5, PREVIOUS_DANCERS = 10, 
                  start_alpha = 0.2, final_alpha = 0.3, 
                  MAX_COUNT = 100):
    
    keypoints = []
    if not os.path.isfile(video_path):
        print(f"ERROR: input video {video_path} could not be found")
        return keypoints

    total_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if MAX_COUNT is None:
        MAX_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer objects
    output_inpainting = init_video_writer(cap, output_fp)
    

    # Process video
    masks = []
    dancers = []
    
    count = 0
    while cap.isOpened() and count < MAX_COUNT:

        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_keypoints, pose_bg, mask = compute_pose_and_mask(frame_rgb)
        keypoints.append([count] + pose_keypoints)

        image = frame_rgb.astype(np.float32) / 255

        # Initiatize painted frame
        black = np.zeros_like(image)
        annotated_image = black.copy()
        
        dancer = None
        if mask is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_size)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_size)
            mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
            dancer = image * mask
            if (count % every_n == 0):
                masks.append(mask)
                dancers.append(dancer)
        else:
            masks.clear()
            dancers.clear()
            
        if len(masks) > PREVIOUS_DANCERS:
            masks = masks[-PREVIOUS_DANCERS:]
            dancers = dancers[-PREVIOUS_DANCERS:]
    
        # Inpaint previous dancers
        alpha = start_alpha
        if len(dancers) > 0:
            alpha_step = start_alpha / len(dancers)
        else:
            alpha_step = 0
            
        for previous_mask,previous_dancer in zip(masks,dancers):
            computed_alpha = alpha - alpha_step * (((count % every_n) / every_n))
            annotated_image = annotated_image * (1 - previous_mask) + \
                                cv2.addWeighted(annotated_image * previous_mask, 1 - computed_alpha, previous_dancer, computed_alpha, 0)
            alpha += alpha_step
        
        # Paint current frame
        if mask is None:
            # annotated_image = image
            annotated_image = black.copy()
        else:
            annotated_image = annotated_image * (1 - mask) + \
                                cv2.addWeighted(annotated_image * mask, 1 - final_alpha, dancer, final_alpha, 0)
            annotated_image = annotated_image * (1 - mask) + dancer
            annotated_image = np.clip(annotated_image, 0, 1)
        annotated_image = (annotated_image * 255).astype(np.uint8)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Write resulting blended frame to the video writer
        output_inpainting.write(annotated_image)
        count += 1

    cap.release()
    output_inpainting.release()

     
    print(f"Number of frames: {MAX_COUNT}")
    print(f"Execution took {(time.time() - total_time):0.2f}.")
    print(f"Compute time per frame: {((time.time() - total_time) / MAX_COUNT):0.4f}.")
    print()

    return keypoints


def process_video_frames(video_path, results_folder, new_width = 360, show_output = False):

    if not os.path.isfile(video_path):
        print("Video not found")
        return

    total_time = time.time()

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height = int(frame_height * new_width / frame_width)

    # Create results folder
    video = video_path.split("/")[-1].split(".")[0]
    results_folder = results_folder + video
    depth_folder = results_folder + "/depth"
    mask_folder = results_folder + "/mask"

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    if not os.path.isdir(depth_folder):
        os.mkdir(depth_folder)
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)


    # Process video
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth = compute_depth(frame)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        if depth.max() > 0:
            depth = (depth / depth.max()) * 255
        depth = depth.astype(np.uint8)

        mask = compute_mask(frame)
        if mask is not None:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            if mask.max() > 0:
                mask = (mask / mask.max()) * 255
            mask = mask.astype(np.uint8)
        else:
            mask = np.zeros(frame.shape)    

        depth = cv2.resize(depth, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))
            
        depth_fp = f"{depth_folder}/{video}_depth_{str(count)}.png"
        mask_fp = f"{mask_folder}/{video}_mask_{str(count)}.png"

        cv2.imwrite(depth_fp, depth)
        cv2.imwrite(mask_fp, mask)

        count += 1

        if show_output:
            output = np.concatenate((mask, depth), axis=1)
            cv2.imshow('Mask and depthmap', output)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of frames: {count}")
    print(f"Execution took {(time.time() - total_time):0.2f}.")
    print(f"Compute time per frame: {((time.time() - total_time) / count):0.4f}.")
    print()

    return depth, mask
