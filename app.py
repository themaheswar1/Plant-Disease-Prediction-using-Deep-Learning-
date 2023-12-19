from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:\FinalDraft.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(120, 120)) # -----------

    """Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0  #Match the rescale factor used during training
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)"""

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    img_array = preprocess_input(img_array)
    # Make prediction
    predictions = model.predict(img_array)

    return predictions

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to temporary
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        class_indices = {'Tomato___Bacterial_spot': 0,
                          'Tomato___Early_blight': 1,
                          'Tomato___Late_blight': 2,
                          'Tomato___Leaf_Mold': 3,
                          'Tomato___Septoria_leaf_spot': 4,
                          'Tomato___Target_Spot': 5,
                          'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 6,
                          'Tomato___Tomato_mosaic_virus': 7,
                          'Tomato___healthy': 8}
        # Make prediction
        predictions = model_predict(file_path, model)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        class_labels = list(class_indices.keys())
        predicted_class_label = class_labels[predicted_class_index]

        return predicted_class_label
    return None

if __name__ == '__main__':
    app.run(debug=True)
