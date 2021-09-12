from flask import Flask, render_template, request
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

lables = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
model = load_model('./models/model1_cifar10.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    imagefile= request.files['imagefile']
    classification = ['','']
    classification[0]= "images/"+ imagefile.filename
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    image1 = cv2.imread(image_path)##loading the image
    
    image1 = cv2.resize(image1, (32, 32))
    image1 = image1 / 255 ##scaling by doing a division of 255
    image1 = img_to_array(image1)

    image1 = np.expand_dims(image1, axis=0) ##expanding the dimensions
    output = model.predict(image1)

    label = np.argmax(output)
    labelName = lables[label]
    classification[1] = "Image is of a " + labelName

    return render_template('index.html', prediction=classification)
