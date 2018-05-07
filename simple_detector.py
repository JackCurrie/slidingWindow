from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.layers import Dropout, Flatten, Dense

import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import imageio as io
import cv2 as cv
import pdb as db


_CAR = 'car'
_NON_CAR = 'non-car'


###################################################
# File Names for model specifications             #
#                                                 #
json_path = './models_json/VGG19_transfer[m1]_3.json'
weights_path = './weights/VGG19_transfer[m1]_3.h5'

# Hardcoded model hyperparameters in for now, may remove
def modelFactory(json_path, weight_path):
    loss = 'categorical_crossentropy'
    num_units_first_dense = 986
    dropout_rate = 0.3867433627389078
    num_units_second_dense = 1043
    lr = 0.006072113068495087
    momentum = 0.7963386502886618

    model_json = open(json_path, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    print('Model loaded from json...')

    l = model.output
    l = Flatten()(l)
    l = Dense(num_units_first_dense, activation='relu')(l)
    l = Dropout(dropout_rate)(l)
    l = Dense(num_units_second_dense, activation='relu')(l)
    final = Dense(2, activation='softmax')(l)
    model = Model(inputs=model.input, outputs=final)
    print('Extra layers of model added')

    model.load_weights(weight_path)
    print('Model weights loaded from disk...')

    model.compile(loss=loss,
                  optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                  metrics=['accuracy'])

    return model


class boxResult:
    x=0
    y=0
    height=0
    width=0
    class_pred=0
    confidence=0.0

    def __init__(self, x, y, height, width, class_pred, confidence):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.class_pred = class_pred
        self.confidence = confidence



def makePredictions(model, image, percent_top_shave, num_bottom_splits):
    MODEL_INPUT_SHAPE = 128

    img_height = image.shape[0]
    img_width = image.shape[1]

    upper_bound = int(img_height * percent_top_shave)
    lower_bound = img_height - 1    # Just to be safe :)

    split_height = lower_bound - upper_bound
    split_width = int(img_width/num_bottom_splits) - 1 # Just to be safe :)

    # Find the width of each of the splits
    # Iterate "num_bottom_splits" times
    #    - Grab the image from the horizontal boundary line down in each of these splits
    #    - Resize it to 128 * 128
    #    - Feed it into the model
    #    - Take the prediction, make a "boxResult" out of it, and append to results list
    results = []
    for tlc_x in range(0, img_width - split_width, split_width):
        print('X: ', tlc_x)
        split = image[upper_bound:(upper_bound + split_height), tlc_x:(tlc_x + split_width), :]

        # For each split, we are going to try to split the img in two vertically,
        # hoping to minimize the distortions of the aspect ratio and whatnot
        split_imgs = []
        split_imgs.append({'x': tlc_x, 'y':upper_bound, 'img': split[0:int((upper_bound+split_height)/2), 0:split_width, :]})
        split_imgs.append({'x':tlc_x, 'y': int(upper_bound+(split_height/2)),  'img': split[int((upper_bound+split_height)/2):, 0:split_width, :]})

        for split_img in split_imgs:
            img = split_img['img']

            scaled_img = cv.resize(img, None, fx=MODEL_INPUT_SHAPE/img.shape[1], fy=MODEL_INPUT_SHAPE/img.shape[0], interpolation=cv.INTER_CUBIC)
            pred = model.predict(np.array([scaled_img,]))
            class_pred = np.argmax(pred)
            result = boxResult(x=split_img['x'], y=split_img['y'], height=int(split_height/len(split_imgs)), width=split_width, class_pred=class_pred, confidence=pred[0][class_pred])
            results.append(result)

    return results


# Here, we are going to make a hacky detector. It will ignore the top parts of images
# (top 1/4 or 1/3), and it will do a "sliding window" 4-6 times across the bottom.
# It might have two layers, I am not sure
#
# Then this program will intake videos, and output the individual frames
#
model = modelFactory(json_path, weights_path)
image_name = './OrderedBusData/image850.jpg'
conf_threshold = .98

image = cv.imread(image_name)
results = makePredictions(model, image, .22, 5)

for res in results:
    if res.class_pred == 0:
        res.class_pred = _CAR
    else:
        res.class_pred = _NON_CAR


print(len(results))


font = cv.FONT_HERSHEY_SIMPLEX
for box in results:
    cv.rectangle(image,(box.x,box.y),(box.x+box.width,box.y+box.height),(255, 0, 0), 2)
    cv.putText(image, box.class_pred + ' ' + str(box.confidence), (box.x, box.y), font, .4, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow('Windows', image)
    cv.waitKey(0)


# Results: This didn't work particularly well, though that is the fault of our detector
#          There seems to be something about the aspect ratio distortion imposed upon the
#          images to fit them into the model that creates many false negatives. 
