import pickle
import cv2
import numpy as np

#Date-----------User--------------------Change-----------
#16/06/2021     Enrique Ramos Garcia    Original creation
#--------------------------------------------------------

#Class PredictEmotions
#Developed in OpenCV and Sk-learn to process and image in numpy array format
#To find up to 10 faces and predict the correspondent emotion for each one of them


class PredictEmotions():
    def __init__(self):
        #Route for Sk-Learn prediction model
        self.predictive_model = 'pickle_Models/modelo_Reco_Facial_LogReg.sav','r'
        #Loading Sk-Learn prediction model with overfitting to dataset FacesDB
        self.predictive_model_Over = 'pickle_Models/modelo_Reco_Facial_LogReg_Over.sav'
        #Route for OpenCV xml face detection haarcascade
        self.face_cascade = cv2.CascadeClassifier('openCV_XMLS/haarcascade_frontalface_alt2.xml')
        #Unpacking saved Sk-Learn model with pickle
        self.loaded_model = pickle.load(open(self.predictive_model[0], 'rb'))
        #Found faces counter
        self.foundFaces = 0
        #Max faces allowed
        self.maxFaces = [''] * 10


    #Method findFacePredictEmotion used to find and predict emotions for up to 10 faces inside the current image
    #Receives an image in 3D numpy array format and returns the same image with a rectangle
    #Surrounding each face and its respective emotion at the bottom of the rectangle
    #Return image is also a 3D numpy array
    def findFacePredictEmotion(self,imageToPredict):
        #Image converted to gray scale color depth
        gray = cv2.cvtColor(imageToPredict, cv2.COLOR_BGR2GRAY)
        #Faces detection with OpenCV
        self.foundFaces = self.face_cascade.detectMultiScale(gray,1.3,5)

        #Traverse and enumerate each face in the picture
        #Every face will have the coordinates surrounding them according to (x,y,w,h)
        for idx, (x,y,w,h) in enumerate(self.foundFaces):
            #Print a rectangle for the current face inside the original image
            cv2.rectangle(imageToPredict, (x,y), (x+w,y+h),(255,0,0),2)
            #Getting just the current face to work with it in the model
            roi_gray = gray[y:y+h,x:x+w]
            #Dimensions used into the SkLearn Model
            dim = (48,48)
            #Resolution reduction for the current face and flattening of the numpy matrix
            #(matrix(48,48) --> vector(2304))
            res = cv2.resize(roi_gray,dim).flatten()
            #Font used on screen/image
            font = cv2.FONT_HERSHEY_SIMPLEX
            #With the 2304 dimensional vector the prediction is done
            self.maxFaces[idx] = self.loaded_model.predict([res])[0]
            #Prediction for the current face is printed in the original image
            cv2.putText(imageToPredict,self.maxFaces[idx],(x, y+h),font, 1,(0, 255, 255),2,cv2.LINE_4)

        return(imageToPredict)


    #Method onlyPredictionString used to find and predict emotions for up to 10 faces inside the current image
    #Receives an image in 3D numpy array format and returns a list with up to 10 strings
    #With the respective emotion for each face
    #Return list will be like: ["Neutral","Happy","Sad","Surprised","Anger","","","","",""]
    def onlyPredictionString(self,image):
        maxFaces = [''] * 10
        count = 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Faces detection with OpenCV
        count = self.face_cascade.detectMultiScale(gray,1.3,5)

        #Traverse and enumerate each face in the picture
        #Every face will have the coordinates surrounding them according to (x,y,w,h)
        for idx, (x,y,w,h) in enumerate(count):
            #Getting just the current face to work with it
            roi_gray = gray[y:y+h,x:x+w]
            #Dimensions used into the SkLearn Model
            dim = (48,48)
            #Resolution reduction for the current face and flattening of the numpy matrix
            #(matrix(48,48) --> vector(2304))
            res = cv2.resize(roi_gray,dim).flatten()
            #With the 2304-dimensional vector the prediction is done
            maxFaces[idx] = self.loaded_model.predict([res])[0]

        return(maxFaces)