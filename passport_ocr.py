import cv2
import pytesseract
import re
from datetime import datetime

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_passport_data(image_path):
    processed = preprocess_image(image_path)

    try:
        text = pytesseract.image_to_string(processed, lang="eng+uzb")
    except Exception as e:
        return {
            "Ism": "Xatolik",
            "Familiya": "Xatolik",
            "Tugilgan sana": "Xatolik",
            "Yosh": "Xatolik",
            "Jins": "Xatolik"
        }

    data = {
        "Ism": "Noma'lum",
        "Familiya": "Noma'lum",
        "Tugilgan sana": "Noma'lum",
        "Yosh": "Noma'lum",
        "Jins": "Noma'lum"
    }

    # Try to extract each field
    name_match = re.search(r"Ism[:\-]?\s*(\w+)", text, re.IGNORECASE)
    surname_match = re.search(r"Familiya[:\-]?\s*(\w+)", text, re.IGNORECASE)
    dob_match = re.search(r"(\d{2}[./\-]\d{2}[./\-]\d{4})", text)
    gender_match = re.search(r"Jins[:\-]?\s*(\w+)", text, re.IGNORECASE)

    if name_match:
        data["Ism"] = name_match.group(1)
    if surname_match:
        data["Familiya"] = surname_match.group(1)
    if dob_match:
        dob = dob_match.group(1).replace('-', '.').replace('/', '.')
        data["Tugilgan sana"] = dob
        try:
            birth_date = datetime.strptime(dob, "%d.%m.%Y")
            age = datetime.today().year - birth_date.year
            data["Yosh"] = age
        except:
            pass
    if gender_match:
        g = gender_match.group(1).capitalize()
        if g in ["Erkak", "Ayol", "Male", "Female"]:
            data["Jins"] = g

    return data

# For testing
if __name__ == "__main__":
    result = extract_passport_data("passport.jpg")
    print(result)
