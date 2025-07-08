import cv2
import os

def maximize_contrast_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def preprocess_image(img_input):
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_contrast = maximize_contrast_clahe(img_gray)
    img_blurred = cv2.GaussianBlur(img_contrast, (5, 5), 0)
    _, img_binary = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binary


output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


input_dir = 'ArUco'

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"cant read {filename}")
            continue

        img_result = preprocess_image(img)

        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.jpg'), img)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_preprocessed.jpg'), img_result)

        print(f"{filename}")

