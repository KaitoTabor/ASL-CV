from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from flask_cors import CORS
import joblib
        


model = joblib.load('asl_model_working.pkl')
scaler = joblib.load('scaler.pkl')
with open('class_names.txt', 'r') as f:
    class_names = f.read().splitlines()



app = Flask(__name__)
CORS(app)
hands = mp.solutions.hands.Hands(static_image_mode=True)

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/')
def ping2():
    return 'pong', 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    image_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hands_result = hands.process(img_rgb)

    if not hands_result.multi_hand_landmarks:
        return jsonify({'sign': 'none'})
    
    landmarks = []
    for lm in hands_result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])


   
    hands_scaled = scaler.transform([landmarks])
    pred_idx = model.predict(hands_scaled)
    pred_class = class_names[pred_idx[0]]
    print(pred_class)

    return jsonify({'sign': pred_class})    

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
