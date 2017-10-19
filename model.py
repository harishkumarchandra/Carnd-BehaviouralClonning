import cv2
import csv
import numpy as np

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def loadImageAndMeasurement(dataPath, imagePath, measurement, images, measurements):
    originalImage = cv2.imread(dataPath + '/' + imagePath.strip())
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*-1.0)

def loadImagesAndMeasurements(dataPath, skipHeader=False, correction=0.2):
    lines = getLinesFromDrivingLogs(dataPath, skipHeader)
    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        # Center
        loadImageAndMeasurement(dataPath, line[0], measurement, images, measurements)
        # Left
        loadImageAndMeasurement(dataPath, line[1], measurement + correction, images, measurements)
        # Right
        loadImageAndMeasurement(dataPath, line[2], measurement - correction, images, measurements)

    return (np.array(images), np.array(measurements))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def trainAndSave(model, inputs, outputs, modelFile, epochs = 3):
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(modelFile)
    print("Model saved at " + modelFile)

def createPreProcessingLayers():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model


def nVidiaModel():
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


print('Loading images')
X_train, y_train = loadImagesAndMeasurements('data', skipHeader=False)
model = nVidiaModel()
print('Training model')
trainAndSave(model, X_train, y_train, 'models/model.h5')
print('The End')
