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

###################################################
# File Names for model specifications             #
#                                                 #
json_path = './models_json/VGG19_transfer[m1]_3.json'
weights_path = './weights/VGG19_transfer[m1]_3.h5'


# Small struct class to hold the bounding boxes and their predictions
class boxResult:
    x=0
    y=0
    height=0
    width=0
    class_pred=0
    confidence=0.0


# Hardcoded model hyperparameters in for now, may remove
def model_factory(json_path, weight_path):
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


# Input: model to use, image to use, top-left-corner coordinates tuple (x, y), and
#        currentScale
# Output: Single boxResult with dimensions scaled back to the un-scaled
#
def modelPredict(model, predictionImage, tlc, currentScale):
    img_height = predictionImage.shape[0]
    img_width = predictionImage.shape[1]

    # !!! I AM NOT SURE IF THIS IS SUFFICIENT/CORRECT PRE-PROCESSING
    #      - Also, if we come back and do some extra aspect ratio stuff!!! This will need to be changed
    #
    predictionImage = np.divide(predictionImage, 255)
    result = model.predict( np.array([predictionImage,]))
    confidence = np.max(result[0])
    class_index = np.argmax(result[0])

    temp = boxResult()
    temp.x = int(tlc[0] / currentScale)
    temp.y = int(tlc[1] / currentScale)
    temp.height = int(img_height / currentScale)
    temp.width = int(img_width / currentScale)
    temp.confidence = confidence

    if class_index == 0:
        temp.class_pred = 'car'
    else:
        temp.class_pred = 'non-car'

    return temp




# Input: Full Image. List of scales to analyze image at. Window size for sliding window
#
#
# Output: Completely unfiltered list of predictions from model
#
#
def get_predictions(model, image, scales, window_size ):
    img_pyramid = []

    for scale in scales:
        scaled_img = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        img_pyramid.append({'scale': scale, 'img': scaled_img})

        # Iterate through scaled_img
        #    Iterate through total number of vertical window strides
        #       Iterate through total number of horizontal window strides
        #          Append prediction with MAPPED coordinates to list
    predictions = []
    for pyr_layer in img_pyramid:
        img_height = pyr_layer['img'].shape[0]
        img_width =  pyr_layer['img'].shape[1]

        tlc_y = 0
        tlc_x = 0
        print('full height/width: ', img_height, ',  ', img_width)

        # !!! Current methodology leaves SLIGHT gaps at bottom of image, revamp later on
        # !!! Additionally, we will need to make adjustments here if we make the aspect ratio update
        #
        if img_height > window_size and img_width > window_size:
            for tlc_y in range(0, (img_height - window_size), int(np.round(window_size * stride_factor))):
                for tlc_x in range(0, (img_width - window_size), int(np.round(window_size * stride_factor))):
                    window_img = pyr_layer['img'][tlc_y:(tlc_y + window_size), tlc_x:(tlc_x + window_size), :]
                    predictions.append(modelPredict(model, window_img, (tlc_x, tlc_y), pyr_layer['scale']))

                # Make a prediction using the farthest-right-possible x value to cover right-side gap
                window_img = pyr_layer['img'][tlc_y:(tlc_y + window_size), (img_width - window_size):img_width, :]
                predictions.append(modelPredict(model, window_img, (img_width - window_size, tlc_y), pyr_layer['scale']))

    return predictions
