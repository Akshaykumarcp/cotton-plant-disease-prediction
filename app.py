from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# define flask app
app = Flask(__name__)

# model name
MODEL_PATH ='resnet152V2_model.h5'

# load trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print('Uploaded image path: ',img_path)
    loaded_image = image.load_img(img_path, target_size=(224, 224))

    # preprocess the image
    loaded_image_in_array = image.img_to_array(loaded_image)

    # normalize
    loaded_image_in_array=loaded_image_in_array/255

    # add additional dim such as to match input dim of the model architecture
    x = np.expand_dims(loaded_image_in_array, axis=0)

    # prediction
    prediction = model.predict(x)

    results=np.argmax(prediction, axis=1)

    if results==0:
        results="The leaf is diseased cotton leaf"
    elif results==1:
        results="The leaf is diseased cotton plant"
    elif results==2:
        results="The leaf is fresh cotton leaf"
    else:
        results="The leaf is fresh cotton plant"

    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(port=5001,debug=True)
