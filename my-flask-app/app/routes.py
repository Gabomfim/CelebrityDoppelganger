import io
import json
import ssl

import numpy as np
import torch
from flask import Blueprint, redirect, render_template, request, url_for
from img2vec_pytorch import Img2Vec
from PIL import Image
from werkzeug.utils import secure_filename

# Configure SSL context
ssl._create_default_https_context = ssl._create_unverified_context

bp = Blueprint('main', __name__)

CELEBRITIES_FOLDER = 'app/celebrities/'
EMBEDDINGS_FILE = 'app/celebrity_embeddings.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

@bp.route('/')
def home():
    return render_template('base.html')

@bp.route('/about')
def about():
    return render_template('about.html')  # Ensure you create about.html in templates directory.

with open(EMBEDDINGS_FILE, 'r') as f:
                precomputed_embeddings = json.load(f)

@bp.route('/upload', methods=['GET', 'POST'])
def find_doppelganger():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image = Image.open(io.BytesIO(file.read()))
            # Process the image here
            # For example, you can convert it to grayscale
            # Initialize Img2Vec
            # Remove the fourth channel if it exists
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            img2vec = Img2Vec(cuda=torch.cuda.is_available())

            # Convert image to vector
            # Convert uploaded image to vector
            uploaded_image_vector = img2vec.get_vec(image)

            # Find the celebrity image with the shortest Euclidean distance
            shortest_distance = float('inf')
            closest_celebrity_image = None

            # Load precomputed embeddings
            for entry in precomputed_embeddings:
                celebrity_image_vector = np.array(entry['embedding'])
                
                # Calculate Euclidean distance
                distance = calculate_euclidean_distance(uploaded_image_vector, celebrity_image_vector)
                
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_celebrity_image = entry['filename']
                    closest_celebrity_name = entry['name']

            # Do something with the processed image
            # For now, we'll just return a success message
            print(f'Closest celebrity image: {closest_celebrity_image} with distance: {shortest_distance}')
            return f'Closest celebrity image: {closest_celebrity_image} with distance: {shortest_distance}'
    return render_template('upload.html')
