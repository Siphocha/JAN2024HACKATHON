import os
from flask import render_template, request, jsonify, send_file
from app import app 

from .model_utils import process_image

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for handling file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser may send an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # If the file is allowed, save it to the upload folder
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process the uploaded image through the segmentation model
        segmented_image_path = process_image(filename)

        # Return the segmented image file
        return send_file(segmented_image_path, mimetype='image/png')

    return jsonify({'error': 'File type not allowed'})

