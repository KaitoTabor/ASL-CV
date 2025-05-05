import cv2
import os

# Get label from user
label = input("Enter the label for this session (e.g., A, B, SPACE): ").strip().upper()
output_dir = os.path.join("captured_images", label)
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 2000

print("Press SPACE to capture an image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Capture - Press SPACE to save", frame)
    key = cv2.waitKey(1)

    # Spacebar pressed: save image
    if key == 32:  # SPACE key
        img_path = os.path.join(output_dir, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    # ESC key: quit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
