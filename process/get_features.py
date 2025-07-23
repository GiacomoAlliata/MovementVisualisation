import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time

# Libs
import torch
import mediapipe as mp

joints = [
 "NOSE",
 "LEFT_EYE_INNER",
 "LEFT_EYE",
 "LEFT_EYE_OUTER",
 "RIGHT_EYE_INNER",
 "RIGHT_EYE",
 "RIGHT_EYE_OUTER",
 "LEFT_EAR",
 "RIGHT_EAR",
 "MOUTH_LEFT",
 "MOUTH_RIGHT",
 "LEFT_SHOULDER",
 "RIGHT_SHOULDER",
 "LEFT_ELBOW",
 "RIGHT_ELBOW",
 "LEFT_WRIST",
 "RIGHT_WRIST",
 "LEFT_PINKY",
 "RIGHT_PINKY",
 "LEFT_INDEX",
 "RIGHT_INDEX",
 "LEFT_THUMB",
 "RIGHT_THUMB",
 "LEFT_HIP",
 "RIGHT_HIP",
 "LEFT_KNEE",
 "RIGHT_KNEE",
 "LEFT_ANKLE",
 "RIGHT_ANKLE",
 "LEFT_HEEL",
 "RIGHT_HEEL",
 "LEFT_FOOT_INDEX",
 "RIGHT_FOOT_INDEX"
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
drawing_stype = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
for info in drawing_stype.values():
    info.thickness = 5
pose = mp_pose.Pose(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8,
                    model_complexity=2,
                    enable_segmentation = True)

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def compute_depth(img):
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

def compute_mask(img, kernel_size = (5, 5)):
    # Process the frame with MediaPipe Pose
    try:
        result = pose.process(img)
    except:
        print("ERROR: pose.process() failed")
        return None
    
    # Draw segmentation mask on the frame
    mask = result.segmentation_mask
    if mask is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_size)

    return mask

def compute_pose_and_mask(img):
    # Process the frame with MediaPipe Pose
    try:
        result = pose.process(img)
    except:
        print("ERROR: pose.process() failed")
        return [[0,0,0,0]] * 33, np.ones_like(img) * 255, np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)

    # Process keypoints
    if result.pose_landmarks is not None:
        keypoints = [[k.x, k.y, k.z, k.visibility] for k in result.pose_landmarks.landmark]
    else:
        keypoints = [[0,0,0,0]] * 33
        
    # Draw the pose landmarks on white background
    annotated_image = np.ones_like(img) * 255
    #annotated_image = np.copy(img)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, 
                                  result.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS,
                                  drawing_stype)
    
    # Draw segmentation mask on the frame
    mask = result.segmentation_mask

    return keypoints, annotated_image, mask

def save_keypoints(keypoints, output_fp):
    data = pd.DataFrame(keypoints, columns = ["frame_index"] + joints)
    data.dropna(inplace = True)
    for joint in joints:
        data[joint] = data[joint].map(lambda x: ",".join([str(w) for w in x]))
    data.to_csv(output_fp, index = False, sep = "\t")
