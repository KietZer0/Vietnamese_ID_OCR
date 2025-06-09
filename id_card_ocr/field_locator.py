# field_locator.py
import cv2
import pytesseract
import numpy as np
from unidecode import unidecode
from .config import FIELD_KEYWORDS, FIELD_ROI_OFFSETS
from .utils import display_image # Assuming utils.py is in the same directory or package

def get_all_text_data(image, lang='vie+eng', config='--psm 11'):
    """Extracts all text data (text, bounding boxes) from an image."""
    try:
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        return data
    except Exception as e:
        print(f"Error in Tesseract: {e}")
        return None

def find_label_box(ocr_data, keywords):
    """Finds the bounding box of the first occurrence of any of the keywords."""
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip().lower()
        # Normalize both keyword and OCR'd word
        normalized_word = unidecode(word)
        for keyword_variant_set in keywords: # keywords is a list of lists/tuples
             for keyword in keyword_variant_set:
                normalized_keyword = unidecode(keyword.lower())
                # Check if OCR'd word contains the keyword part
                # This is a simple check; might need more sophisticated matching
                if normalized_keyword in normalized_word:
                    # Could also try to match sequences of words for multi-word labels
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                                  ocr_data['width'][i], ocr_data['height'][i])
                    return (x, y, w, h) # Return the first match
    return None


def get_field_value_roi(warped_card_image, label_box, roi_offset_params, debug=False):
    """
    Calculates the ROI for a field's value based on its label's bounding box
    and offset parameters.
    roi_offset_params: (y_offset_factor, x_offset_factor_from_label_end, width_factor_of_card, height_multiplier_of_label)
    """
    card_h, card_w = warped_card_image.shape[:2]
    l_x, l_y, l_w, l_h = label_box
    
    y_offset_factor, x_offset_factor, width_factor, height_factor = roi_offset_params

    # Value ROI y relative to label y
    val_y = int(l_y + y_offset_factor * l_h)
    
    # Value ROI x relative to label's end
    val_x = int(l_x + l_w + x_offset_factor * l_w)
    
    # Value ROI width as a factor of card width
    val_w = int(width_factor * card_w)
    if val_x + val_w > card_w: # Ensure it doesn't go out of bounds
        val_w = card_w - val_x
        
    # Value ROI height as a multiplier of label height
    val_h = int(l_h * height_factor)

    # Ensure ROIs are within image bounds
    val_y = max(0, val_y)
    val_x = max(0, val_x)
    val_h = min(val_h, card_h - val_y)
    val_w = min(val_w, card_w - val_x)

    if val_h <=0 or val_w <=0:
        print(f"Warning: Calculated ROI for value has zero or negative dimension: y:{val_y} x:{val_x} h:{val_h} w:{val_w}")
        return None

    value_roi_bbox = (val_x, val_y, val_w, val_h)

    if debug:
        debug_img = warped_card_image.copy()
        cv2.rectangle(debug_img, (l_x, l_y), (l_x + l_w, l_y + l_h), (0, 255, 0), 1) # Label in green
        cv2.rectangle(debug_img, (val_x, val_y), (val_x + val_w, val_y + val_h), (255, 0, 0), 1) # Value in blue
        display_image(f"Label_Value_ROI", cv2.resize(debug_img, (0,0), fx=0.5, fy=0.5))

    return warped_card_image[val_y:val_y + val_h, val_x:val_x + val_w]


def locate_all_fields(warped_card_image, debug=False):
    """Locates all predefined fields on the warped ID card image."""
    field_rois = {}
    ocr_data = get_all_text_data(warped_card_image)

    if not ocr_data:
        print("OCR data could not be extracted from the warped card.")
        return field_rois

    for field_name, keywords_for_field in FIELD_KEYWORDS.items():
        label_bbox = find_label_box(ocr_data, [keywords_for_field]) # find_label_box expects a list of keyword lists
        if label_bbox:
            roi_params = FIELD_ROI_OFFSETS.get(field_name)
            if roi_params:
                value_image_roi = get_field_value_roi(warped_card_image, label_bbox, roi_params, debug=debug)
                if value_image_roi is not None and value_image_roi.size > 0 :
                    field_rois[field_name] = value_image_roi
                else:
                    print(f"Could not extract ROI for {field_name} value, or ROI is empty.")
            else:
                print(f"ROI offset parameters not defined for {field_name}")
        else:
            print(f"Could not find label for {field_name} using keywords: {keywords_for_field}")
            if debug: # If label not found, show full card for this field
                field_rois[field_name] = warped_card_image 
                
    return field_rois