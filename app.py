from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from flask_cors import CORS
import joblib
        


model = joblib.load('asl_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('class_names.txt', 'r') as f:
    class_names = f.read().splitlines()



app = Flask(__name__)
CORS(app)
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/')
def ping2():
    return 'pong', 200


def is_hand_open(landmarks):
    return (
        landmarks[8].y < landmarks[6].y and
        landmarks[12].y < landmarks[10].y and
        landmarks[16].y < landmarks[14].y and
        landmarks[20].y < landmarks[18].y
    )

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

    lefthand = None
    righthand = None

    if(hands_result.multi_hand_landmarks is None):
        return jsonify({'sign': 'none'})


    for index, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
        handedness = hands_result.multi_handedness[index].classification[0].label

        #webcam flip
        if handedness == 'Right':
            lefthand = hand_landmarks
        elif handedness == 'Left':
            righthand = hand_landmarks
            
    if not righthand:
        return jsonify({'sign': 'none'})
    
    landmarks = []
    for lm in righthand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])


   
    hands_scaled = scaler.transform([landmarks])
    pred_idx = model.predict(hands_scaled)
    pred_class = class_names[pred_idx[0]]
    print(pred_class)
    if lefthand:
        open = is_hand_open(lefthand.landmark)
    else:
        open = False

    return jsonify({'sign': pred_class,'open': open})    

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
