# %%
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ImageClassification:
    def __init__(self):
        self.IMAGE_SIZE = 28
        self.IMAGES_PATH = 'images/train'
        self.IMAGES_PATH_PREDICT = 'images/predict'
        self.IMAGE_LABEL_DATASET_PATH = 'image_label_dataset.csv'
        self.LABELS = os.listdir(self.IMAGES_PATH)
        self.INPUT_X_Y = []
        self.MODEL = None

    # To initiate the process
    def start(self, retrain=True):
        if(retrain):
            try:
                # Remove the input csv file, so it will be freshly created
                os.remove(self.IMAGE_LABEL_DATASET_PATH)
            except IOError:
                None
            self.read_images()
            self.prepare_data()
            self.train_model()
        else:
            # Load existing trained model file
            if(os.path.exists('image_classification_model.h5')):
                self.MODEL = keras.models.load_model(
                    'image_classification_model.h5')
            else:
                print('image_classification_model.h5 : file not found')

    def prepare_data(self):
        # Read the csv data as dataframe (rows,columns)
        ds = pd.read_csv(self.IMAGE_LABEL_DATASET_PATH,
                         header=None)
        ds = ds.sample(frac=1)  # Shuffle the Rows so the data is mixed up, easy for train test data split

        # Define x : Input/Feature Data
        x = ds.iloc[:, :(ds.shape[1]-1)].values 
        # Define y : Output/Label/Class Data
        y = ds.iloc[:, -1].values

        # map the string data to integer data
        yle = LabelEncoder() 
        y = yle.fit_transform(y) 
        
        # convert the data to categorical format
        y = keras.utils.np_utils.to_categorical(y)
        self.LABELS = yle.classes_ # keep the Labels Data to reverse map the predicted data
        print('LABELS : ', self.LABELS)

        # Split Data to Train and Test set , so one set used to Train the Model and rest is used to Test and compare the Result (Accuracy)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=.2, random_state=42)
        self.INPUT_X_Y = [x_train, x_test, y_train, y_test]

    def train_model(self):
        x_train, x_test, y_train, y_test = self.INPUT_X_Y

        # Here is actul Keras CNN Model is created to Train on the Data and Predictions

        model = Sequential()
        # Input Layer
        model.add(Conv2D(32, (3, 3), input_shape=(
            self.IMAGE_SIZE, self.IMAGE_SIZE, 1), activation='relu'))
        
        # Hidden Layers
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(len(self.LABELS), activation='softmax'))

        # Compile the Model
        model.compile(optimizer='adadelta',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        # format the x input data to the format supported by the model.
        # Since its CNN Model it expects a Matrix format where we reshape 
        # the each x array (1,784) to Matrix format (28,28,1)
        x_train = x_train.reshape(
            x_train.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        x_test = x_test.reshape(
            x_test.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

        # Train the model 
        # We have 10 labels and 10 images per label so Total 10*10 = 100 input rows
        # since we have less input data, we eed to specify few hunderd iteratuion to get better accuracy
        # so we try 2000 times to extract key features for btter prediction
        # too much or too less iterations impact the accuracy 
        model.fit(x_train, y_train, epochs=2000, batch_size=32, verbose=0)
        loss, accuracy = model.evaluate(x_test, y_test)

        print('loss : ', loss)
        print('accuracy : ', accuracy * 100) # model accuracy 0 - 100. >70 is better
        # save the model so we can directly predict without need to train everytime
        model.save('image_classification_model.h5') 
        self.MODEL = model

    def predict(self, path):
        if(path == None):
            path = '/2.png'
        image_path = self.IMAGES_PATH_PREDICT+path
        pred = self.process_prediction(image_path)

        # simply predict the image is what type of image (Label the image correctly)
        print('image: ', image_path, ' >> prediction : ', pred)

    def process_prediction(self, image_path):
        # map the predicted data to the label
        image = self.image_to_input(image_path)
        pred = self.MODEL.predict(image) # get the predicted data which is in array of probabilities which sums up to 1
        pred = np.argmax(pred, axis=1) # get the index of the array which has most probability
        return self.LABELS[pred[0]] # map the index to the label

    def image_to_input(self, image_path):
        # convert image file into x input format, so given to the model to predict
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        return image.reshape((1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))

    def read_images(self):
        # iterate through the folders where each folder is taken as the Label and 
        # images inside will be taken as the input data
        for dir in os.listdir(self.IMAGES_PATH):
            for file in os.listdir(self.IMAGES_PATH+'/'+dir):
                self.image_to_csv(self.IMAGES_PATH+'/'+dir+'/'+file, dir)

    def image_to_csv(self, image_path, label):
        # read the image file and convert it into input data format(e.g: table data) and
        # append it to the csv file
        image = cv2.imread(image_path) # Read image file
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert into grayscale image
        image = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE)) # resize image to 28x28
        image = image.flatten() # format the 28x28 matrix into 1x784 matrix 
        image = np.append(image, label).reshape(
            1, self.IMAGE_SIZE*self.IMAGE_SIZE+1) # Append the Label name as the ast column
        pd.DataFrame(image).to_csv(self.IMAGE_LABEL_DATASET_PATH,
                                   mode='a', header=False, index=False) # Append the row data to the csv file


# To use CPU / GPU for tensorflow operations 
with tf.device('/gpu:0'): # use '/cpu:0' in case of no gpu available
    imageClassification = ImageClassification()
    
    # True : To Re-Train Model with New set of Images and Lables
    # False : To Load existing Trained Model and simply predict
    retrain = False
    if(retrain):
        imageClassification.start(retrain=True)
    else:
        imageClassification.start(retrain=False)
    
    # Predict the Labels of given images
    imageClassification.predict(path='/0.png')
    imageClassification.predict(path='/1.png')
    imageClassification.predict(path='/2.png')
    imageClassification.predict(path='/3.png')
    imageClassification.predict(path='/4.png')
    imageClassification.predict(path='/5.png')
    imageClassification.predict(path='/6.png')
    imageClassification.predict(path='/7.png')
    imageClassification.predict(path='/8.png')
    imageClassification.predict(path='/9.png')
    imageClassification.predict(path='/10.png')

keras.backend.clear_session()    


# %%
