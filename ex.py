from flask import Flask, request, send_file
import subprocess
import os
from flask_cors import CORS
import requests
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    print('Received text:', text)
    print('Received headers:', request.headers)

    if not text:
        return {'error': 'No text provided'}, 400

    # Define file paths
    pose_file = 'output.pose'
    gif_file = 'output.gif'

    # Run the text_to_gloss_to_pose command
    command = [
        'text_to_gloss_to_pose',
        '--text', text,
        '--glosser', 'simple',
        '--lexicon', 'assets/dummy_lexicon',
        '--spoken-language', 'en',
        '--signed-language', 'ase',
        '--pose', pose_file
    ]
    result = subprocess.run(command, cwd='spoken-to-signed-translation', capture_output=True, text=True)

    if result.returncode != 0:
        return {'error': 'Translation failed', 'details': result.stderr}, 500

    # Visualize the pose and save as GIF
    from pose_format import Pose
    from pose_format.pose_visualizer import PoseVisualizer

    with open(os.path.join('spoken-to-signed-translation', pose_file), 'rb') as f:
        pose = Pose.read(f.read())
        
    # Resize to 256, for visualization speed
    scale = pose.header.dimensions.width /500
    pose.header.dimensions.width = int(pose.header.dimensions.width/scale)
    pose.header.dimensions.height = int(pose.header.dimensions.height/scale)
    pose.body.data = pose.body.data / scale

    visualizer = PoseVisualizer(pose)
    visualizer.save_gif(gif_file, visualizer.draw())
    print("GIF saved to:", gif_file)
    # Upload the GIF to IPFS
   
    
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    print("printing api key: ",os.getenv("PINATA_API_KEY"))
    headers = {
        "pinata_api_key": os.getenv("PINATA_API_KEY"),  # Load from .env
        "pinata_secret_api_key": os.getenv("PINATA_SECRET_API_KEY")  # Load from .env
    }
    # Open your file in binary mode
    with open("output.gif", "rb") as file:
        files = {"file": file}
        response = requests.post(url, headers=headers, files=files)
    # Serve the GIF file
    if response.status_code == 200:
     print("Upload successful:", response.json())
     return response.json(),200
    else:
     print("Upload failed:", response.text)
     return {'error': 'Upload failed', 'details': response.text}, 500

if __name__ == '__main__':
    app.run(port=5002)
