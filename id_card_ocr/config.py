# config.py

# Keywords to identify labels (include common OCR variations if known)
# (Vietnamese, English, Vietnamese without diacritics)
FIELD_KEYWORDS = {
    "id_number": ["Số", "No.", "So"],
    "full_name": ["Họ và tên", "Full name", "Ho va ten"],
    "dob": ["Ngày sinh", "Date of birth", "Ngay sinh"],
    "sex": ["Giới tính", "Sex", "Gioi tinh"],
    "nationality": ["Quốc tịch", "Nationality", "Quoc tich"],
    "place_of_origin": ["Quê quán", "Place of origin", "Que quan"],
    "place_of_residence": ["Nơi thường trú", "Place of residence", "Noi thuong tru"],
    "date_of_expiry": ["Có giá trị đến", "Date of expiry", "Co gia tri den"]
}

# Relative ROI for the *value* once a label is found.
# (dx_factor, dy_factor, width_factor, height_factor) relative to label's bbox
# (x, y, w, h) of the label.
# The value ROI will be:
# x_val = x_label + dx_factor * w_label
# y_val = y_label + dy_factor * h_label
# w_val = width_factor * w_card (or some other reference)
# h_val = height_factor * h_label
# These will need careful tuning based on the card layout.
# Using a simpler model for now: (y_offset, x_offset_from_label_end, width_guess, height_multiplier)
# y_offset: how much to shift y from label's y
# x_offset_from_label_end: how much to the right of the label's end to start the value ROI
# width_guess: how wide the value ROI should be (can be a large portion of remaining card width)
# height_multiplier: value_roi_height = label_height * height_multiplier
FIELD_ROI_OFFSETS = {
    "id_number":          (0, 0.1, 0.5, 1.5), # y_offset_factor, x_offset_factor, w_factor, h_factor (all relative to label)
    "full_name":          (0, 0.1, 0.8, 1.5), # These need to be tuned carefully
    "dob":                (0, 0.1, 0.4, 1.5),
    "sex":                (0, 0.1, 0.2, 1.5),
    "nationality":        (0, 0.1, 0.3, 1.5), # Nationality is often short
    "place_of_origin":    (1.0, -0.5, 1.0, 3.0), # Below and slightly left of label, wide, multi-line
    "place_of_residence": (1.0, -0.5, 1.0, 3.0), # Below and slightly left of label, wide, multi-line
    "date_of_expiry":     (0, 0.1, 0.4, 1.5)
}

# Tesseract configurations per field
# PSM modes:
# 6: Assume a single uniform block of text.
# 7: Treat the image as a single text line.
# 8: Treat the image as a single word.
# 11: Sparse text. Find as much text as possible in no particular order.
# 13: Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
TESSERACT_FIELD_CONFIG = {
    "id_number": "--psm 7 -c tessedit_char_whitelist=0123456789",
    "full_name": "--psm 7",
    "dob": "--psm 7 -c tessedit_char_whitelist=0123456789/",
    "sex": "--psm 7",
    "nationality": "--psm 7",
    "place_of_origin": "--psm 6", # For multi-line text
    "place_of_residence": "--psm 6", # For multi-line text
    "date_of_expiry": "--psm 7 -c tessedit_char_whitelist=0123456789/"
}

# For face detection (relative to card dimensions after warping)
# (ymin_factor, xmin_factor, ymax_factor, xmax_factor)
FACE_ROI_FACTORS = (0.20, 0.05, 0.70, 0.30) # Approximate from image