import cv2
import os

import os

# Ensure the paths work regardless of where the script is invoked from
current_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(current_dir, "..", "Data", "Seeds")
os.makedirs(save_folder, exist_ok=True)
cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Define central ROI boundaries (Yellow Box limits)
    # E.g. A 224x224 cutout in the middle of a standard 640x480 webcam feed
    h, w = frame.shape[:2]
    crop_size = 224
    
    # Calculate top-left and bottom-right coords to center the box
    start_x = w // 2 - crop_size // 2
    start_y = h // 2 - crop_size // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    
    # Draw Yellow Bounding Box (BGR: 0, 255, 255) of thickness 2
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    
    cv2.imshow("Seed Capture (Press 'S' to Save, 'Q' to Quit)", display_frame)

    key = cv2.waitKey(1)

    # press S to save seed
    if key == ord('s'):
        # Crop the exact region inside the yellow box to save (Matching the 224x224 Model input)
        roi = frame[start_y:end_y, start_x:end_x]
        
        filename = os.path.join(save_folder, f"seed_{count}.jpg")
        cv2.imwrite(filename, roi)

        print("Saved:", filename)

        count += 1

    # press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()