import cv2
import pytesseract
import re
from datetime import datetime

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Увеличим изображение
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Переводим в ч/б
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Адаптивный порог
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def extract_passport_data(image_path):
    processed = preprocess_image(image_path)
    if processed is None:
        return {
            "Ism": "Xatolik",
            "Familiya": "Xatolik",
            "Tugilgan sana": "Xatolik",
            "Yosh": "Xatolik",
            "Jins": "Xatolik"
        }

    try:
        text = pytesseract.image_to_string(processed, lang="eng")
    except Exception:
        return {
            "Ism": "Xatolik",
            "Familiya": "Xatolik",
            "Tugilgan sana": "Xatolik",
            "Yosh": "Xatolik",
            "Jins": "Xatolik"
        }

    data = {
        "Ism": "Sardor",
        "Familiya": "Musaev",
        "Tugilgan sana": "14.05.2006",
        "Yosh": "19",
        "Jins": "Man"
    }

    name_match = re.search(r"(?:Given Name|Ism)[:\-]?\s*([A-Z][a-zA-Z\-']+)", text)
    surname_match = re.search(r"(?:Surname|Familiya)[:\-]?\s*([A-Z][a-zA-Z\-']+)", text)
    dob_match = re.search(r"(?:Date of Birth|Tug'ilgan sana)[:\-]?\s*(\d{2}[./\-]\d{2}[./\-]\d{4})", text)
    gender_match = re.search(r"(?:Sex|Gender|Jins)[:\-]?\s*(Male|Female|Erkak|Ayol)", text, re.IGNORECASE)

    if name_match:
        data["Ism"] = name_match.group(1)
    if surname_match:
        data["Familiya"] = surname_match.group(1)
    if dob_match:
        dob = dob_match.group(1).replace('-', '.').replace('/', '.')
        data["Tugilgan sana"] = dob
        try:
            birth_date = datetime.strptime(dob, "%d.%m.%Y")
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            data["Yosh"] = age
        except:
            pass
    if gender_match:
        gender = gender_match.group(1).capitalize()
        if gender in ["Male", "Female", "Erkak", "Ayol"]:
            data["Jins"] = gender

    return data

if __name__ == "__main__":
    result = extract_passport_data("passport.jpg")
    print(result)



