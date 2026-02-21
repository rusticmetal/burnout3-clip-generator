import cv2
import torch
from torchvision import transforms
import imageio_ffmpeg
import subprocess
import os
import argparse

from burnoutCNN import *

'''
Run this after you successfully created a .pth file with high enough validation accuracy. This script will find the moments with takedowns 
and create clips under ./clips/.
'''

parser = argparse.ArgumentParser(description="Burnout Takedown Clips Generator")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--format", required=True, help="Output clips extension (not including the dot, must be supported by ffmpeg)")
parser.add_argument("--model", required=True, help="Model .pth file")
args = parser.parse_args()

video_path = args.input
video_extension = args.format
pth = args.model
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

print("Loading CNN from: " + pth)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BurnoutCNN().to(device)
model.load_state_dict(torch.load(pth, map_location=device))
model.eval()

#same transform as training the cnn
transform_list = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
transform = transforms.Compose(transform_list)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
last_detection_frame = -1
min_frame_gap = fps * 2.5 #takedown camera lasts about 2.5 seconds, so don't capture another takedown for this amount of time
takedown_times = []

print("Now scanning video for takedown moments.")
while True:
    retval, frame = cap.read()
    if not retval:
        break
    
    #turn the frame into a tensor
    frame_resize = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    #now we can try to predict if the frame is a takedown
    with torch.no_grad():
        output = model(frame_tensor)
        probability = torch.softmax(output, dim=1) #one probability for each of the 2 classes, normalized from tensor values
        confidence, predicted_class = torch.max(probability, 1)

    if predicted_class.item() == 1 and confidence.item() > 0.8: #we are confident this frame is a takedown moment
        if frame_index - last_detection_frame > min_frame_gap:
            time_sec = frame_index / fps
            print(f"Takedown detected at {time_sec:.2f} seconds")
            takedown_times.append(time_sec)
            last_detection_frame = frame_index
    frame_index += 1

cap.release()

print("Now starting takedown clip generation.")
#wasn't 100% sure about how much time to have before/after the takedown moment, but these values seem fine
before_seconds = 8
after_seconds = 3

for i, t in enumerate(takedown_times):
    start_time = max(t - before_seconds, 0)
    output_file = os.path.join(CLIPS_FOLDER, "clip_" + str(i + 1) + video_extension)

    #write the clip file
    cmd = [ffmpeg_path, "-y", "-ss", str(start_time), "-i", video_path, "-t", str(before_seconds + after_seconds), "-c", "copy", output_file]
    subprocess.run(cmd)
    print(f"Saved {output_file}")

print("Finished, check the " + video_path + " directory.")