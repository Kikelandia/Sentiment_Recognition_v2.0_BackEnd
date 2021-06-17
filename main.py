from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from imageProcessing.convertVideoImage import ConvertFormat
from dataPrediction.predictor import PredictEmotions

#Date-----------User--------------------Change-----------
#16/06/2021     Enrique Ramos Garcia    Original creation
#--------------------------------------------------------

#App to implement Sk Learn model for facial prediction, developed by Enrique Ramos García
#Developed with OpenCV to find faces and predict emotions on them
#It can predict between: Neutral, Sad, Happy, Surprised, Anger
#And with a maximum of 10 faces inside the video/picture
#Web sockets implemented in order to reach a better/faster response between the back/front end

app= Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

#Route to test if app loads correctly
@app.route('/', methods=['GET'])
def index():
    return "<h1>Predict Index</h1>"

#Instances of the classes in charge of convert images and predict emotions
conversor = ConvertFormat()
predictor = PredictEmotions()

#Web socket to receive a complete image and predict the facial expression
#To work correctly, function must receive a JPEG image, and will return a JPEG image
#with each face surrounded by a rectangle and the predicted emotion to the bottom of each rectangle
@socketio.on('fullImage')
def fullImage(data_image):
#conversor = ConvertFormat()
#predictor = PredictEmotions()
    #Image conversion from base64 to 3D numpy array
    openCvImage = conversor.fromBytesToCV2(data_image)
    #3D Array analyzed to predict emotions, final image created
    imgWithPredEmotions = predictor.findFacePredictEmotion(openCvImage)
    #Final image encoded back to base64 format
    finalBytesStr = conversor.fromNpArrToBytes(imgWithPredEmotions)
    #Finally emitting the image to render in the front-end
    emit('response_back', finalBytesStr[2:-1])


#Web socket to receive a complete image and predict the facial expression
#To work correctly, function must receive a JPEG image, and will return an array
#With the strings for each facial expression, max 10 faces
@socketio.on('tensorFlowImage')
def tensorFlowImage(data_image):
#conversor = ConvertFormat()
#predictor = PredictEmotions()
    #Image conversion from base64 to 3D numpy array
    openCvImage = conversor.fromBytesToCV2(data_image)
    #Predict emotion for each face and get list with all of them
    emotionsArray = predictor.onlyPredictionString(openCvImage)

    emit('predsList', emotionsArray)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1')