import cv2
import numpy as np
import pandas as pd

from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# === File Paths ===
mask_path = r"C:\Users\sande\OneDrive\Desktop\cvproject\parking-space-counter\mask_1920_1080.png"
video_path = r"C:\Users\sande\OneDrive\Desktop\cvproject\data\parking_1920_1080_loop.mp4"

# === Load Mask ===
mask = cv2.imread(mask_path, 0)
if mask is None:
    print("Error: Failed to load mask image. Check the file path.")
    exit()

# === Load Video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Failed to open video file. Check the file path.")
    exit()

# === Get Parking Slot Bounding Boxes ===
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# === Initialization ===
spots_status = [False for _ in spots]
diffs = [0 for _ in spots]
previous_frame = None
frame_nmr = 0
ret = True
step = 30

# === Process Video ===
while ret:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, (x1, y1, w, h) in enumerate(spots):
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            prev_crop = previous_frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, prev_crop)

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            max_diff = np.amax(diffs) if np.amax(diffs) > 0 else 1
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / max_diff > 0.4]

        for spot_indx in arr_:
            x1, y1, w, h = spots[spot_indx]
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, (x1, y1, w, h) in enumerate(spots):
        color = (0, 255, 0) if spots_status[spot_indx] else (0, 0, 255)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Text Box and Status
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    available_count = sum(1 for s in spots_status if s)
    total_spots = len(spots_status)
    cv2.putText(frame, f'Available spots: {available_count} / {total_spots}', (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

# === Save Summary to CSV ===
total_slots = len(spots_status)
occupied_slots = sum(1 for s in spots_status if not s)
available_slots = sum(1 for s in spots_status if s)

data = {
    "Total Number of Slots": [total_slots],
    "Occupied Slots": [occupied_slots],
    "Available Slots": [available_slots]
}

df = pd.DataFrame(data)
df.to_csv('parking_slots_summary.csv', index=False)
print("CSV output saved to parking_slots_summary.csv")
