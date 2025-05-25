import streamlit as st
import cv2
import numpy as np
import time

ZONES = {
    "Zone 1": ((100, 100), (300, 300)),
    "Zone 2": ((350, 100), (550, 300)),
    "Zone 3": ((100, 350), (300, 550)),
    "Zone 4": ((350, 350), (550, 550))
}

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

st.set_page_config(layout="wide")
st.title("🛰️ Yuqoridan Kamera: Mijoz Kuzatuv Tizimi")
st.warning("Bu demo RTSP emas, lokal kamera bilan ishlaydi. RTSP uchun URL bilan almashtiring.")

run_camera = st.checkbox("📹 Ishga tushurish")
frame_placeholder = st.empty()
log_placeholder = st.empty()

def load_model():
    net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def get_outputs(net):
    ln = net.getLayerNames()
    return [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_people(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    H, W = frame.shape[:2]
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == 0 and confidence > CONF_THRESHOLD: 
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    people = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        people.append((x, y, x + w, y + h))
    return people

def determine_zone(center):
    for zone, (top_left, bottom_right) in ZONES.items():
        if top_left[0] <= center[0] <= bottom_right[0] and top_left[1] <= center[1] <= bottom_right[1]:
            return zone
    return None

if run_camera:
    cap = cv2.VideoCapture(0)
    net = load_model()
    output_layers = get_outputs(net)
    zone_logs = {key: 0 for key in ZONES.keys()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detect_people(frame, net, output_layers)

        for label, (pt1, pt2) in ZONES.items():
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for (x1, y1, x2, y2) in people:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            zone = determine_zone((cx, cy))
            if zone:
                zone_logs[zone] += 1
                cv2.putText(frame, zone, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        log_placeholder.write(zone_logs)

    cap.release()
