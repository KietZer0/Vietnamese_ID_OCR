# text_extractor.py
import pytesseract
import re
from unidecode import unidecode
from .config import TESSERACT_FIELD_CONFIG
from .utils import preprocess_for_ocr

def ocr_field_roi(image_roi, field_name, lang='vie'):
    """Performs OCR on a given image ROI for a specific field."""
    if image_roi is None or image_roi.size == 0:
        return ""
        
    preprocessed_roi = preprocess_for_ocr(image_roi)
    
    config = TESSERACT_FIELD_CONFIG.get(field_name, "--psm 7") # Default to PSM 7
    try:
        text = pytesseract.image_to_string(preprocessed_roi, lang=lang, config=config)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR for field {field_name}: {e}")
        return ""

def clean_id_number(text):
    return "".join(re.findall(r'\d', text))

def clean_name(text):
    # Remove extra spaces, potential OCR noise specific to names
    text = re.sub(r'\s+', ' ', text).strip()
    # Example: Capitalize each word, though Tesseract often does this
    # For Vietnamese names, this is more complex due to multi-part last/middle names
    # A simple title case might not be perfect but is a start.
    # return ' '.join(word.capitalize() for word in text.split())
    # Given the example's output, it seems to be all caps for names.
    # Let's try to keep it as Tesseract outputs mostly, just clean whitespace.
    # The provided example uses a common_last_name.txt, which is a good idea.
    
    # Basic cleaning
    cleaned_text = re.sub(r'[^\w\sÀ-ỹA-Z]', '', text) # Keep alphanumeric, whitespace, and Vietnamese characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Heuristic: If text is mostly uppercase, keep it. If mixed, uppercase it.
    # Or simply uppercase everything for consistency like the ID.
    return cleaned_text.upper()


def clean_dob(text):
    # Try to find DD/MM/YYYY or DD.MM.YYYY
    match = re.search(r'(\d{1,2})[/\.](\d{1,2})[/\.](\d{4})', text)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"
    # If not found, try to extract 8 digits and format them
    digits = "".join(re.findall(r'\d', text))
    if len(digits) == 8:
        return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
    return text # Return raw if no specific format found

def clean_sex(text):
    text_lower = unidecode(text.lower())
    if "nam" in text_lower or "male" in text_lower:
        return "Nam"
    if "nu" in text_lower or "female" in text_lower:
        return "Nữ"
    return text # Return raw if unsure

def clean_nationality(text):
    text_lower = unidecode(text.lower())
    if "viet nam" in text_lower:
        return "Việt Nam"
    return text.strip()

def clean_address(text):
    # Addresses can be complex. Basic cleaning for now.
    text = re.sub(r'\s+', ' ', text).strip(": ")
    return text.replace('\n', ', ') # Replace newlines with commas for readability

def extract_and_clean_field(image_roi, field_name, lang='vie'):
    raw_text = ocr_field_roi(image_roi, field_name, lang=lang)
    
    if field_name == "id_number":
        return clean_id_number(raw_text)
    elif field_name == "full_name":
        return clean_name(raw_text)
    elif field_name == "dob":
        return clean_dob(raw_text)
    elif field_name == "sex":
        return clean_sex(raw_text)
    elif field_name == "nationality":
        return clean_nationality(raw_text)
    elif field_name in ["place_of_origin", "place_of_residence", "date_of_expiry"]:
        return clean_address(raw_text) # Using generic address cleaning for now
    return raw_text