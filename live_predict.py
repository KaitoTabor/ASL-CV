import cv2
import numpy as np
import mediapipe as mp
import joblib


model = joblib.load('asl_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('class_names.txt', 'r') as f:
    class_names = f.read().splitlines()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape((21, 3))
    wrist = landmarks[0]
    landmarks -= wrist
    max_dist = np.linalg.norm(landmarks, axis=1).max()
    if max_dist > 0:
        landmarks /= max_dist
    return landmarks.flatten()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640, 640))

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    sign = 'none'
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]

        mp_drawing.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
            # landmarks.extend([lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z])

        if len(landmarks) == 63:
            # normed = normalize_landmarks(landmarks)
            # print(f"Normalized landmarks: {normed}")
            # scaled = scaler.transform([normed])
            scaled = scaler.transform([landmarks])
            probs = model.predict_proba(scaled)
            conf = np.max(probs)
            if conf >= 0.6:
                pred_class = class_names[np.argmax(probs)]
                sign = f"{pred_class} ({conf:.2f})"

    
    # frame = cv2.resize(frame, (640, 480))
    cv2.putText(frame, f"Detected: {sign}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
