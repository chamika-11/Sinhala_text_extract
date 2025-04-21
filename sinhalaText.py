import os
# Manually set TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = r"C:/Program Files/Tesseract-OCR/tessdata/"
import numpy as np
import cv2
import pytesseract
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# # Manually set the path to tessdata
custom_oem_psm_config = '--oem 3 --psm 6'

image_path = "id_card_1742376877.jpg"
image = cv2.imread(image_path)

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




# Apply contrast enhancement
alpha = 1.5   # Contrast control (1.0-3.0)
beta = 1     # Brightness control (0-100)
gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# #Resize the image
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# #Apply denoising to image
gray = cv2.GaussianBlur(gray, (5,5),0)
    
#adaptive thresholder
# gray =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#morphological transformation(Text enhancement)
kernel = np.ones((1, 1), np.uint8)
gray=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)





# Save the preprocessed image
cv2.imwrite("Extracted_image.jpg", gray)

pil_image = Image.fromarray(gray)

sinhala_text = pytesseract.image_to_string(pil_image, lang='sin', config=custom_oem_psm_config)

print("Extracted Sinhala Text:")
print(sinhala_text)
