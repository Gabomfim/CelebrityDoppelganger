import os
import json
from img2vec_pytorch import Img2Vec
from PIL import Image

# Initialize Img2Vec
img2vec = Img2Vec()

# Directory containing celebrity images
celebrities_folder = 'celebrities'

# Output JSON file
output_file = 'celebrity_embeddings.json'

# List to hold JSON entries
json_entries = []

# Iterate over each image in the celebrities folder
for filename in os.listdir(celebrities_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Extract celebrity name from filename
        celebrity_name = ' '.join(filename.split('_'))
        
        # Load image
        img_path = os.path.join(celebrities_folder, filename)
        img = Image.open(img_path)
        
        # Get image embedding
        embedding = img2vec.get_vec(img).tolist()
        
        # Create JSON entry
        json_entry = {
            'filename': filename,
            'celebrity_name': celebrity_name,
            'embedding': embedding
        }
        
        # Add entry to list
        json_entries.append(json_entry)

# Write JSON entries to file
with open(output_file, 'w') as f:
    json.dump(json_entries, f, indent=4)