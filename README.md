# Burnout 3 Takedown Clip Generator

This is mostly a recreation of this tutorial: https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html using snapshots from Burnout 3: Takedown gameplay, to get a CNN to recognize whenever the player scores a takedown on an enemy car. The scripts included can break down videos into normal and takedown frames for the CNN to train upon, train the CNN with a binary classification (takedown occurring vs. normal gameplay), and extract 10-15 second takedown clips from a gameplay video using a trained model.

<h2>Important</h2>

You need the takedown camera enabled in your game settings.

A working model was trained on about 35 minutes of gameplay footage (at 60fps, with one frame snapshot being taken every 5 frames).

The .pth file for the model was not included because it was 25+ megabytes.

Run `pip install -r requirements.txt` first to make sure the modules are installed.

By default, the data folder is ./data/ and the clips folder is ./clips/, relatively referenced from the scripts in this directory. Make sure ./data/normal/ and ./data/takedown/ exist, and specify --input and --output locations if you are using different folder paths (be sure to change the paths in `burnoutCNN.py`)

<h2>Instructions to recreate the CNN and create clips</h2>

1. Record 30+ minutes of Burnout 3 gameplay footage (Road Rage mode at 60fps is optimal).

2. Run `generate_frames.py` on your gameplay video file. Make sure that ./data/normal/ exists so that the frames can be created there.

3. Run `move_frames.py`, and watch your gameplay video at 2-3x speed. Pause whenever a takedown occurs (ideally when the boost meter appears and gets filled) and input the timestamp (in the format 'Minutes;Seconds', with a semicolon) to the python script. This will move all nearby frames around that time to the /data/takedown/ folder. Alternatively, you could try automating this but it really only takes 10-15 minutes.

4. Now you can create and train the CNN, by running `train_burnout_cnn.py`. Make sure to enter a name for the .pth file at the end.

5. Now (assuming your gameplay was sufficient to train the model, verify the correctness percent of the validation set if you are unsure) whenever you have more gameplay and want to see all takedown moments, you can run `takedown_clip_generator.py` on any new gameplay videos and your takedown clips will be created under /clips/. (Note: Running this can rewrite clips that are currently in the directory, so move the ones you care about)