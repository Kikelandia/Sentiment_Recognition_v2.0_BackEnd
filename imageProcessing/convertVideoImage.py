import base64
import cv2
import numpy as np

#Date-----------User--------------------Change-----------
#16/06/2021     Enrique Ramos Garcia    Original creation
#--------------------------------------------------------

#Class ConvertFormat
#Developed to convert/transform between image formats

class ConvertFormat():


    #Method fromBytesToCV2 used to convert from a bytes like string to a numpy array
    #In order to convert correctly between data types string should start with
    #data:image/jpeg;base64, iVBORw0KGgo..., else will throw an exception
    #Return is a numpy array
    def fromBytesToCV2(self, bytesStrToConvert):
        #Base64 binary to text decoding
        decoded_data = base64.b64decode(bytesStrToConvert[23:])
        #Text to numpy codification
        np_data = np.fromstring(decoded_data, dtype=np.uint8)
        #Unchanged image as Open CV numpy array
        npArray = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        return(npArray)


    #Method fromNpArrToBytes used to convert image from numpy array to bytes like string
    #Should receive a OpenCV numpy array, a normal numpy array might create unknown behavior
    #Return is a string with data:image/jpeg;base64, iVBORw0KGgo..., format
    def fromNpArrToBytes(self, imageAsNumpyArray):
        #Encoding image from numpy array to bytes like string
        retval, stringData = cv2.imencode('.jpg', imageAsNumpyArray)
        #Base 64 encoding
        jpgImageAsString = base64.b64encode(stringData)

        return(str(jpgImageAsString))