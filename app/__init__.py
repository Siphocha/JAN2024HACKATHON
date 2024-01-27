from flask import Flask
import os

app = Flask(__name__)
current_directory = os.path.dirname(os.path.realpath(__file__))

uploads_folder = os.path.join(current_directory, 'uploads')

app.config['UPLOAD_FOLDER'] = uploads_folder

from app import routes


app.run(debug=True)