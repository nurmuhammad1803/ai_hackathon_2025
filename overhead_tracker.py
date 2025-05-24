import cv2
import numpy as np
from ultralytics import YOLO
import time

# RTSP Stream URL (replace with actual one)
RTSP_URL = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' model is lightweight, fast

# Define zones (4 demo zones as rectangles)
ZONES = {
    "🚗 Malibu": [(100, 100), (300, 300)],
    "🚙 Cobalt": [(350, 100), (550, 300)],
    "🛋️ Waiting": [(100, 350), (300, 550)],
    "🚘 Tracker": [(350, 350), (550, 550)]
}

# Helper to check if a point is inside a zone rectangle
def point_in_zone(point, zone):
    (x1, y1), (x2, y2) = zone
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open RTSP stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame.")
            break

        results = model(frame, classes=[0], verbose=False)[0]  # Class 0 = person

        people_positions = []

        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = map(int, det[:6])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            people_positions.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw zones and check occupancy
        for label, ((x1, y1), (x2, y2)) in ZONES.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        zone_counts = {label: 0 for label in ZONES}
        for (cx, cy) in people_positions:
            for label, rect in ZONES.items():
                if point_in_zone((cx, cy), rect):
                    zone_counts[label] += 1

        # Display counts on frame
        y_offset = 30
        for label, count in zone_counts.items():
            text = f"{label}: {count} person(s)"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        cv2.imshow("Overhead Zone Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
