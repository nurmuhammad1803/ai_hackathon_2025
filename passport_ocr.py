import cv2
import pytesseract
import numpy as np


def debug_ocr(image_path):
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    cv2.imwrite("debug_processed.jpg", thresh)
    print("[INFO] Saved B&W processed image to debug_processed.jpg")

    print("[INFO] Running pytesseract...")
    text = pytesseract.image_to_string(thresh)
    print("\n===== OCR TEXT OUTPUT =====")
    print(text.strip())
    print("===========================\n")

    print("[INFO] Detected character boxes:")
    boxes = pytesseract.image_to_boxes(thresh)
    print(boxes.strip())

if __name__ == "__main__":
    # Replace with your passport image filename
    img_path = "passport.jpg"
    debug_ocr(img_path)