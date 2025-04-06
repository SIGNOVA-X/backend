# SIGNOVA-X BACKEND Documentation

Welcome to the **SIGNOVA-X Backend** repository!  
This repository hosts the Python server codebase powering the **SIGNOVA-X** Flutter application – part of the SUPANOVA platform.

---

## 🚀 Overview

### 🔧 `server.py`
Main backend server script for the SUPANOVA application.  
It supports:
- Real-time translation between **sign language ↔ spoken language**.
- AI-based **profile recommendation**.
- **Emergency response** functionality.

### 🧠 `recommender.py`
A **Simple AI-Based Profile Matching System**:
- Takes user data (e.g. age, job, interests, language) from a JSON file.
- Preprocesses and converts it into numerical feature vectors.
- Uses **cosine similarity** to recommend similar users based on shared attributes.

### 💬 `motivationalChatbot.py`
Builds a **motivational chatbot** using **Google Gemini (Generative AI)** API:
- Accepts user prompts.
- Responds with motivational messages using a fine-tuned LLM.

### ✋ `SignToText.py`
Handles **Sign Language to Text Conversion** using:
- A **trained CNN model**.
- Hand tracking landmarks.
- Accepts image/video frame inputs.
- Predicts real-time alphabet or command based on hand features.

---

## 🌐 API Endpoints & Features

### 1. 🔄 Text to Sign Language GIF Translation
**`POST /translate`**
- Accepts spoken text.
- Converts it to sign language pose sequences using a subprocess.
- Visualizes the poses into a GIF of an avatar performing sign language.
- Uploads the GIF to **IPFS** using **Pinata API**.
- Returns the **IPFS hash** of the uploaded GIF.

---

### 2. 🎥 Video to Image Frame Extraction  
**`POST /convert-image`**
- Accepts a sign language video.
- Extracts one frame every 4 seconds.
- Saves them as individual images for prediction.

---

### 3. 🤖 Sign Prediction from Images  
**`GET /predict-all-images`**
- Scans all images from `output_images/` directory.
- Uses CNN model to classify each sign.
- Returns predictions for every frame.

---

### 4. 🖼️ Serve Translated GIF  
**`GET /get-gif`**
- Serves the generated sign language GIF for frontend playback.

---

### 5. 🧑‍🤝‍🧑 Friend Recommendation  
**`POST /recommend`**
- Accepts a `username`.
- Returns a list of **recommended friends** based on profile similarities.
- Uses the `SimpleProfileRecommender` and data from `data.json`.

---

## 🧠 How the Recommender Works

The `SimpleProfileRecommender` class is responsible for recommending users with similar profiles using a weighted feature similarity approach.

### 🔄 User Data Preprocessing
- **Age Normalization** using `MinMaxScaler`.
- **Categorical Encoding** for fields like `job`, `sex`, and `language` using one-hot encoding.
- **Interest Vectorization** using **TF-IDF** to convert text-based interests into numerical vectors.

### ⚖️ Weighted Profile Matching
- Assigns specific **weights** to each feature (e.g., age, job, interests) to control their influence on matching.
- Applies a **diagonal weight matrix** to emphasize fields like interests more heavily.

### 📈 Similarity Calculation
- Calculates **cosine similarity** between the current user and all others.
- Returns the **top 5 most similar users**, excluding the current user.

### 📬 Recommendation Output
- Based on similarity scores, returns a list of **recommended usernames** who closely align with the user's preferences and attributes.

---

## ✋ How SignToText Works

The **SignToText** module translates hand gestures into English letters using a combination of landmark detection and CNN-based classification.

### 🔍 Input & Detection
- Accepts input as an **image or frame**.
- Detects **hand landmarks** using `cvzone`’s `HandDetector`.
- Refines gesture prediction using **landmark logic** and a **trained CNN model**.

### 🧠 `predict_gesture(...)`
- Core logic that:
  - Takes in **hand landmarks (`pts`)** and a **processed hand image (`test_image`)**.
  - Uses CNN to predict gesture groups (e.g., `Group 0`, `Group 1`, etc.).
  - Applies **custom landmark-based rules** to improve accuracy and convert gesture groups into actual **English letters** (e.g., `A`, `B`, `T`).

### 🖼️ `predict_from_file(image_path)`
- Designed to predict signs from **static images**.
- Steps:
  - Detects the hand in the image.
  - Draws landmarks on a blank canvas.
  - Passes the canvas to the CNN model for prediction.
  - Returns the **predicted alphabet** for frontend display or avatar rendering.

---

## 🧠 About the Model

### 🧹 Data Preprocessing and Feature Extraction

The hand sign recognition pipeline begins with **webcam-based image capture**, followed by hand detection using the **MediaPipe** library — a powerful tool for real-time hand tracking and landmark extraction.

Once the hand is detected:
- The **Region of Interest (ROI)** is cropped from the frame.
- The ROI is converted to **grayscale** using OpenCV to simplify image data.
- A **Gaussian blur** is applied to reduce noise and smooth the image.
- **Thresholding and adaptive thresholding** are used to binarize the grayscale image, clearly separating the hand gesture from the background.

For training, a diverse dataset of hand gestures representing the **English alphabet (A–Z)** was collected, including variations in angles and hand shapes to enhance robustness.

![Hand landmark diagram showing 21 key points labeled from WRIST (0) to PINKY_TIP (20), with red dots and green connecting lines representing finger joints.](assets/hand_Node.png)

---

### 🧠 Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)** forms the core of the sign language classification model. CNNs are well-suited for computer vision tasks like image recognition and classification due to their ability to learn spatial hierarchies of features.

#### CNN Architecture Overview:
- **Convolutional Layers**: Extract low- to high-level visual features such as edges, curves, and textures.
- **Pooling Layers** (e.g., Max Pooling): Reduce spatial dimensions and retain important features.
- **Flatten Layer**: Converts multi-dimensional feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers used for the final classification.
- **Dropout Layers**: Applied to prevent overfitting during training.

![Convolutional Neural Network architecture showing input image patch, two hidden layers with feature maps, max pooling, and final classification layer with 4 output units.](assets/cnn.png)


Unlike traditional neural networks, CNNs:
- Use 3D neuron arrangements (width, height, depth).
- Focus on **local receptive fields**, meaning each neuron processes a small patch of the previous layer — reducing computation and improving efficiency.

#### Output:
- The final output layer returns a **vector of class probabilities** across all sign language classes (`A` to `Z`).
- The **highest probability class** represents the model's predicted gesture.

---

## 🛠️ Tech Stack & Features Used

### 🚀 Backend Framework
- **Flask** – Lightweight web framework used to build RESTful APIs for communication between frontend and backend.
- **CORS** – Enables Cross-Origin Resource Sharing, allowing frontend apps to make requests to the backend server.

### 🧠 AI/ML & Model Integration
- **Keras** – Used for loading and running the trained CNN model for gesture prediction.
- **scikit-learn**:
  - `MinMaxScaler`, `OneHotEncoder`, `TfidfVectorizer`, `cosine_similarity` – For preprocessing and user similarity calculation.
  - **PCA** (assumed from recommender.py) – For dimensionality reduction and clustering in recommendations.
- **scipy.sparse** – Efficient handling of large, sparse matrices used in text and profile vectorization.

### 🖼️ Image & Video Processing
- **OpenCV (`cv2`)** – For image preprocessing, hand ROI cropping, and thresholding.
- **imageio** & **PIL** – For extracting frames from videos and saving them as images.
- **MediaPipe** – Used to detect hand landmarks in real-time for sign language recognition.
- **cvzone.HandTrackingModule** – Simplifies MediaPipe integration and extracts 21 hand landmarks.

### 🎨 Visualization & Animation
- **Pose Format / PoseVisualizer** – Used to visualize sign language poses as animations and generate sign GIFs.

### ☁️ Storage & API Integration
- **Pinata/IPFS API** – Uploads the translated GIFs to decentralized storage, returning an IPFS hash.
- **Werkzeug** – Handles secure file transfers in Flask for uploaded media files.

### 💬 Generative AI
- **google.generativeai** – Integrates with Google Gemini to power the motivational chatbot with dynamic, AI-generated responses.

### 📊 Data Handling
- **Pandas** – Loads and manipulates user profile data for recommender systems.
- **NumPy** – Performs numerical computations, array transformations, and vector operations.

---

## 🚀 Usage

Follow these steps to set up and run the SIGNOVA-X Backend locally:

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/SIGNOVA-X/backend.git
cd backend
```

## 🛠️ Environment Setup (.env)

To run the project locally, create a `.env` file in the root directory and add the following environment variables:
create a account on pinata.cloud and generate the api from here [Pinata](https://pinata.cloud/)
```env
# 📦 add your Pinata (IPFS) API Keys

PINATA_API_KEY=""
PINATA_SECRET_API_KEY=""

```
Clone the Repository

```bash
git clone  https://github.com/ZurichNLP/spoken-to-signed-translation
cd spoken-to-signed-translation
pip install .
```

## 🌐 Ngrok Setup for Backend

### 🔧 Step-by-Step Setup

1. **Install Ngrok**

   If you don’t have Ngrok installed, run:

   ```bash
   # For macOS
   brew install ngrok/ngrok/ngrok
   ```
   ```bash
   # For Ubuntu/Debian
   sudo snap install ngrok
    ```
   ### Or download manually from: https://ngrok.com/download

   ```bash
   ngrok config add-authtoken <YOUR_AUTH_TOKEN>
   ```
   Inside your terminal run:

   ```bash
    ngrok http 5002
   ```
   This will generate a url like this one for you : NGROK_URL="https://<your-id>.ngrok-free.app"

### 🐍 2. Create and Activate a Virtual Environment

#### 🔹 On macOS/Linux:

```bash
python3.10 -m venv venv
source venv/bin/activate
```
#### 🔹 On Windows:

```bash
python3.10 -m venv venv
venv/Scripts/activate
```
### 📦 3. Install Dependencies

Make sure you have `pip` installed. Then run:

```bash
pip install -r requirements.txt
```

### ▶️ 4. Run the Backend Server
```bash
python3.10 server.py
```
If everything is set up correctly, you should see:
```bash
 * Running on http://127.0.0.1:5002/ (Press CTRL+C to quit)
```
To deactivate the virtual environment
```bash
deactivate
```


