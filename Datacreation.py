import cv2
import os

label = input("Enter the label for this session (e.g., A, B, SPACE): ").strip().upper()
output_dir = os.path.join("captured_images", label)
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 3000

print("Press SPACE to capture an image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - Press SPACE to save", frame)
    key = cv2.waitKey(1)

    if key == 32: 
        img_path = os.path.join(output_dir, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
