import pytesseract
import cv2
import numpy as np

def find_receipt_contour(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assumed to be the receipt)
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the maximum contour
    x, y, w, h = cv2.boundingRect(max_contour)

    return x, y, w, h
def preprocess_receipt(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Find the bounding box of the receipt
    x, y, w, h = find_receipt_contour(image)

    # Crop the image to the receipt region
    cropped_image = image[y:y + h, x:x + w]

    # Resize the image to a reasonable size
    resized_image = cv2.resize(cropped_image, (800, 600))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Further noise reduction using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

    return grayscale_image

# # Example usage:
# input_image_path = "path/to/your/receipt/photo.jpg"
# preprocessed_image = preprocess_receipt(input_image_path)
#
# # Display the original and pre-processed images
# cv2.imshow("Original Image", cv2.imread(input_image_path))
# cv2.imshow("Preprocessed Image", preprocessed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def extract_text_from_image(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text


def extract_items_from_text(text):
    items = []

    # Split text into lines
    lines = text.split('\n')

    # Extract description and price from each line
    for line in lines:
        # Example: "Item description $9.99"
        match = re.match(r'(.+)\s+([\d.,]+)', line)
        if match:
            description, price = match.groups()
            items.append({"description": description.strip(), "price": price.strip()})

    return items


# # Example usage:
# input_image_path = "path/to/your/receipt/photo.jpg"
# preprocessed_image = preprocess_receipt(input_image_path)
#
# # Extract text from the preprocessed image
# extracted_text = extract_text_from_image(preprocessed_image)
#
# # Extract description and price of each item
# items = extract_items_from_text(extracted_text)
#
# # Display the extracted items
# print("Extracted Items:")
# for item in items:
#     print(f"Description: {item['description']}, Price: {item['price']}")
#
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = preprocess_receipt('/home/rob/Pictures/grocery_receipts/IMG-20231203-WA0009.jpg')
    rwh = extract_text_from_image(img)
    print(rwh)


