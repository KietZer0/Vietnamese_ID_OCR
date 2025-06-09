# main.py
import cv2
import argparse
from .card_preprocessor import crop_and_warp_card, extract_face_roi
from .field_locator import locate_all_fields
from .text_extractor import extract_and_clean_field
from .config import FACE_ROI_FACTORS
from .utils import display_image # Assuming utils.py is in the same directory or package


def main(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # 1. Detect and warp the ID card
    warped_card = crop_and_warp_card(image, debug=debug)
    if warped_card is None:
        print("ID card could not be processed.")
        return
    
    if debug:
        display_image("Final Warped Card", cv2.resize(warped_card, (0,0), fx=0.7, fy=0.7))

    # 2. Extract Face ROI (optional, but common)
    face_image = extract_face_roi(warped_card, FACE_ROI_FACTORS)
    if debug and face_image is not None:
        display_image("Face ROI", face_image)
        # cv2.imwrite("output/face.jpg", face_image)


    # 3. Locate ROIs for each text field
    # Increase resolution for better OCR data for field location
    scale_factor = 2.0 # Example scale factor
    h, w = warped_card.shape[:2]
    hi_res_card = cv2.resize(warped_card, (int(w*scale_factor), int(h*scale_factor)), 
                             interpolation=cv2.INTER_CUBIC)
    
    field_rois = locate_all_fields(hi_res_card, debug=debug) # Pass hi_res_card here
                                                         # ROIs returned will be from hi_res_card

    # 4. Perform OCR on each field ROI and clean the text
    extracted_data = {}
    for field_name, roi_image in field_rois.items():
        if roi_image is not None and roi_image.size > 0:
            # ROIs from locate_all_fields are from hi_res_card, no need to pass original warped_card
            text = extract_and_clean_field(roi_image, field_name)
            extracted_data[field_name] = text
            if debug:
                print(f"Field: {field_name}, Raw ROI shape: {roi_image.shape}")
                display_image(f"ROI for {field_name}", roi_image)
        else:
            extracted_data[field_name] = "N/A (ROI not found or empty)"


    # 5. Print results
    print("\n--- Extracted ID Card Information ---")
    for field, value in extracted_data.items():
        print(f"{field.replace('_', ' ').title()}: {value}")
    print("------------------------------------")

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR for Vietnamese Citizen ID Card.")
    parser.add_argument("-i", "--image", required=True, help="Path to the ID card image.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (display intermediate images).")
    args = parser.parse_args()

    # You might need to set TESSDATA_PREFIX environment variable
    # export TESSDATA_PREFIX=/path/to/your/tessdata/
    # Or configure tesseract path in script if needed by pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS

    main(args.image, args.debug)