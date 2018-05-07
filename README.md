# Sliding Window
Hello, and welcome the "Sliding Window" repository. The is where we take the model that we have trained,
and surround it with the necessary components to have a full object detector pipeline. Here I will
give a quick overview of files/directories here.


### models_json directory
Keras models are saved and loaded using two components: a json file with the model's specifications,
and a weights file containing the actual weights in each of the filters and the fully connected layers
in the network. This dir contains the json.


### weights
Contains the weights, as discussed above. But the files are pretty big (> 150Mb), so they cannot be uploaded to Github.
If you are interested in actually trying this out, then contact me at `jackcurrie@protonmail.com` and I can send you the weights.


### simple_detector.py
Here, we create a highly simplified version of the object detector. We use a valid assumption that vehicles are not going to be in the top part of the image, and ignore the top 20-30% of each frame, and then we simply take full-height rectangle (4-6) of them from the bottom part of the screen, resize them, and feed them into the model. The results were not promising for this method, as these large slices of the image are very different from the training and validation data (where vehicles were individually cropped out, and all non-vehicular images were squares).


### analyze_frame.py
We Made a dumb, single-scaled sliding window detector here, which doesn't have a conclusively working non-max suppression algorithm in it yet. We lined it up to intake entire directories of images, analyze them, and then spit out the analyzed images
into a single directory. Then we used `ffmpeg` to turn this directory into a video.
[This is the video](https://www.youtube.com/watch?v=nay1rOBJdCA)


### adjust_dir.py
Don't worry about this one. While taking the images through ffmpeg to make the video, it required that we had a sequence
of image files with names which contained an ordered sequence of integers. Somehow this got misaligned, so this is a little bandaid script to resequnce the images.


### The Notebooks
The jupyter notebooks named `sliding_window.ipynb` and `sliding_window_v2.0.ipynb` are essentially just sketch books where we played around with the code that would later on be put into the analyze_frame program. If you would like to view their contents, just run `jupyter notebook` on your system and you will be able to open and run them.
