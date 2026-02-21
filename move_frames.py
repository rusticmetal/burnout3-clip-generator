import os
import shutil
import argparse

from burnoutCNN import *

'''
Use this after generating frames. You should enter timestamps manually with a semicolon as "Minutes;Seconds" e.x "17:09", while watching the gameplay video at 2-3x speed 
(or you could automate but idk). Try to include aftertouches but not deaths. You can try to include takedowns that do not 
have a takedown camera (happens when many takedowns happen in succession) but I didn't.
'''

parser = argparse.ArgumentParser(description="Moves takedown frames from the normal directory to the takedown directory. Timestamps must be manually submitted.")
parser.add_argument("--input", default=os.path.join(DATA_FOLDER, "/normal/"), help="Normal frames directory")
parser.add_argument("--output", default=os.path.join(DATA_FOLDER, "/takedown/"), help="Takedown frames directory")
args = parser.parse_args()
NORMAL_DIRECTORY_PATH = args.input
TAKEDOWN_DIRECTORY_PATH = args.output

os.makedirs(TAKEDOWN_DIRECTORY_PATH, exist_ok=True)
FRAMES = os.listdir(NORMAL_DIRECTORY_PATH)
TIME_WINDOW = 0.5 #this is the time window (both before and after, in seconds) in which all frames matching the timestamp will move (so its a total 1 second window during the takedown)

input_time = None
while (input_time != ""):
    input_time = input("Enter timestamp (Minutes;Seconds): ")
    try:
        minutes = float(input_time.split(";")[0])
        seconds = float(input_time.split(";")[1])

        for file in FRAMES:
            file_time = file.split("_t")[1].replace(".jpg", "")
            frame_time = float(file_time)

            #check to see if the frame time is within the the window to be eligible to count as a takedown frame, then move it over
            if abs(frame_time - ((minutes * 60) + seconds)) <= TIME_WINDOW:
                shutil.move(os.path.join(NORMAL_DIRECTORY_PATH, file), os.path.join(TAKEDOWN_DIRECTORY_PATH, file))
                print("Moved frame at " + str(frame_time) + " to " + TAKEDOWN_DIRECTORY_PATH) 
    except:
        print("Ending.")