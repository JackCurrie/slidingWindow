from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.layers import Dropout, Flatten, Dense

import numpy as np
import os
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


# Input: model to use to analyze window. [model]
#        Image.[image]
#        List of scales to analyze image at.[scales]
#        Window size for sliding window [window_size]
#
#
# Output: Completely unfiltered list of predictions from model
#
#
def getPredictions(model, image, scales, window_size):
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
        # !!! Additionally, we will need to make adjustments here if we make
        #     decide to include different non-square aspect ratios
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

#
#
def nonMaxSuppression(predictions, conf_thresh, overlap_thresh):
    filtered_preds = [pred for pred in predictions if pred.class_pred == 'car' and pred.confidence > conf_thresh]
    if not filtered_preds:
        return []

    # (x1, y1) is the top-left corners
    # (x2, y2) are the bottom-right corners
    x1 = np.array([pred.x for pred in filtered_preds])
    y1 = np.array([pred.y for pred in filtered_preds])
    x2 = np.array([pred.x + pred.width for pred in filtered_preds])
    y2 = np.array([pred.y + pred.height for pred in filtered_preds])


    # Get vector of areas, find indices of  boxes in order of theikr "height" location on the image
    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(y2)

    # Keep looping while some indices still remain in the indices list
    pick = []
    while len(indices) > 0:
        last = len(indices) - 1
        index = indices[last]
        pick.append(indices[-1])

        # Find largest (x, y) coordinates for the start of the bounding box and
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[index], x1[indices[:last]])
        yy1 = np.maximum(y1[index], y1[indices[:last]])
        xx2 = np.minimum(x2[index], x2[indices[:last]])
        yy2 = np.minimum(y2[index], y2[indices[:last]])

        # Tutorial mentions something about  a  "+ 1", but I don't really get it
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / areas[indices[:last]]
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # print(predictions)
    return [pred for i, pred in enumerate(filtered_preds) if i in pick]



# Now all of the methods are defikned. We will first just make this program
# output the detections for a single image, then refactor to be inside of a
# class so that the model doesn't need to reload every time that an image is analyzed
#
out_dir = './out_sliding_75stride_95thresh/'
data_dir = './OrderedBusData/'

window_size = 128
stride_factor = .75
scales = [1]
conf_threshold = .92
overlap_threshold = .3  # typical values between .3 and .5 according to internet

model = modelFactory(json_path, weights_path)

img_names = os.listdir(data_dir)

for image_name in img_names:
    image = cv.imread(data_dir + image_name)
    predictions = getPredictions(model, image, scales, window_size)

    suppressed_preds = nonMaxSuppression(predictions, conf_threshold, overlap_threshold)
    just_car_preds = [pred for pred in predictions if pred.class_pred == 'car' and pred.confidence > conf_threshold]

    font = cv.FONT_HERSHEY_SIMPLEX

    for box in suppressed_preds:
        cv.rectangle(image,(box.x,box.y),(box.x+box.width,box.y+box.height),(255, 0, 0), 2)
        cv.putText(image, box.class_pred + ' ' + str(box.confidence), (box.x, box.y), font, .4, (255, 255, 255), 1, cv.LINE_AA)
        cv.imwrite(out_dir + image_name, image)

    # cv.imshow('Windows', image)
    # cv.waitKey(0)


# cv.destroyAllWindows()





#
