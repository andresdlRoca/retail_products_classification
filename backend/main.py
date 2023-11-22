import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained models from the .h5 files
model1 = load_model('./backend/model1.h5')
model2 = load_model('./backend/model2.h5')
model3 = load_model('./backend/model3.h5')
model4 = load_model('./backend/model4.h5')

# Define a route for predicting with model1
@app.route('/predict/model1', methods=['POST'])
def predict_model1():
    try:
        # Get the image data from the request
        image_data = request.files['image'].read()
        
        # Inside your predict_modelX functions:
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((715, 700))  # Resize the image
        img_array = image.img_to_array(img)  # Convert to numpy array

        # Expand dimensions to add batch_size
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions with model1
        predictions = model1.predict(img_array)
        
        # Format the predictions as a JSON response
        response = {
            'predictions': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for predicting with model2
@app.route('/predict/model2', methods=['POST'])
def predict_model2():
    try:
        # Get the image data from the request
        image_data = request.files['image'].read()
        
        # Inside your predict_modelX functions:
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((715, 700))  # Resize the image
        img_array = image.img_to_array(img)  # Convert to numpy array

        # Expand dimensions to add batch_size
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions with model1
        predictions = model2.predict(img_array)
        
        # Format the predictions as a JSON response
        response = {
            'predictions': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for predicting with model3
@app.route('/predict/model3', methods=['POST'])
def predict_model3():
    try:
        # Get the image data from the request
        image_data = request.files['image'].read()
        
        # Inside your predict_modelX functions:
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((715, 700))  # Resize the image
        img_array = image.img_to_array(img)  # Convert to numpy array

        # Expand dimensions to add batch_size
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions with model1
        predictions = model3.predict(img_array)
        
        # Format the predictions as a JSON response
        response = {
            'predictions': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Define a route for predicting with model4
@app.route('/predict/model4', methods=['POST'])
def predict_model4():
    try:
        # Get the image data from the request
        image_data = request.files['image'].read()
        
        # Inside your predict_modelX functions:
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((715, 700))  # Resize the image
        img_array = image.img_to_array(img)  # Convert to numpy array

        # Expand dimensions to add batch_size
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions with model1
        predictions = model4.predict(img_array)
        
        # Format the predictions as a JSON response
        response = {
            'predictions': predictions.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
