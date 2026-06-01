import cv2
import numpy as np
import pytesseract
from pydantic import BaseModel

text_recognition_title="Text Recognition"
text_recognition_description="This model performs Optical Character Recognition (OCR) to extract text from images."

class OCRPredictIn(BaseModel):
    x: str

class OCRPredictOut(BaseModel):
    data: str

class ChunkedOCRRequestIn(BaseModel):
    x: str
    request_id: str
    packet_index: int
    total_packets: int

class ChunkedOCRRequestOut(BaseModel):
    data: str


custom_config = r'--oem 3 --psm 6'  # OEM: 3=LSTM only, PSM: 6=assume single uniform block of text

def ocr_preprocessing(image_str: str) -> np.ndarray:
    """Preprocessing function for OCR using pytesseract.

    Args:
        image_str (str): 1D vector of uint8 image in string format

    Returns:
        np.ndarray: preprocessed grayscale image
    """
    # Load 1d vector of uint8 image from string
    img = np.array([int(i) for i in image_str.split(",")], dtype=np.uint8)
    if img.size != 60*180:
        return None, "Invalid image size. Data must be comma separated and with a shape 1x10800"

    # Reshape
    img = img.reshape((60, 180))  # Assuming the image size is known (e.g., 60x180)
    return img, None

def ocr_predict(input_data: np.ndarray) -> str:
    text = pytesseract.image_to_string(input_data, config=custom_config)
    text = text.replace("\n", "")
    return text