import cv2
import os
import argparse

from burnoutCNN import *

'''
Use this first on your gameplay video(s) (30+ minutes ideally) to generate the frames needed to feed to the model. 
After running this you need to also run move_frames to putthe takedown frames separate in their own data folder. 
After that can you run train_burnout_cnn.py.
'''

parser = argparse.ArgumentParser(description="Generates frames from gameplay.")
parser.add_argument("--input", required=True, help="Input video path")
parser.add_argument("--output", default=os.path.join(DATA_FOLDER, "/normal/"), help="Output frames (normal) directory")
args = parser.parse_args()
VIDEO_PATH = args.input
OUTPUT_DIRECTORY =  args.output
FRAME_SKIP = 5

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
saved_frames = 0

while True:
    retval, frame = cap.read()
    if not retval:
        break

    #write every 5th frame with its number and timestamp to the data folder
    if frame_index % FRAME_SKIP == 0:
        timestamp_seconds = frame_index / fps
        frame = cv2.resize(frame, (224, 224))
        filename = os.path.join(OUTPUT_DIRECTORY, f"frame_{saved_count:06d}_t{timestamp_seconds:.2f}.jpg") #pad the beginning with zeroes to make sure the images are viewed in order
        cv2.imwrite(filename, frame)
        print("Wrote: " + filename)
        saved_count += 1
    frame_index += 1

cap.release()
print(f"Saved " + str(saved_frames) + " frames.")