import cv2
import os

video_folder = "Data/videos"
output_folder = "Data/Frames"

os.makedirs(output_folder, exist_ok=True)

# Number of frames to extract per second
target_fps = 2

# Supported video extensions
valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}

# Get only video files
video_files = [f for f in os.listdir(video_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

print(f"Found {len(video_files)} video files.")

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}. Skipping.")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: FPS is 0 for {video_file}. Skipping.")
        cap.release()
        continue
    interval = max(1, int(fps / target_fps))

    frame_id = 0
    saved_id = 0
    
    video_name = os.path.splitext(video_file)[0]
    print(f"Processing {video_file} (FPS: {fps:.2f}, Interval: {interval})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            filename = f"{video_name}_{saved_id:04d}.jpg"
            path = os.path.join(output_folder, filename)

            cv2.imwrite(path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()
    print(f" -> Extracted {saved_id} frames from {video_file}.")

print("Frame extraction complete.")