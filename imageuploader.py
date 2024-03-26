import streamlit as st
from PIL import Image
import io
import easyocr
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
import cv2

# Initialize the EasyOCR reader
reader = easyocr.Reader(["en"])

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs.input_ids,
        num_beams=4,
        min_length=30,
        max_length=150,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def perform_ocr_and_summarization(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Perform OCR on the thresholded image
    result = reader.readtext(threshold_image)

    # Extract text from the OCR result
    text = " ".join([res[1] for res in result])

    # Generate summary
    summary = generate_summary(text)

    # Display the OCR result and summarized text
    st.title("Text extracted from the image:")
    st.write(text)

    # Put a box around the summarized text

    st.title("Summarized text:")
    st.write(summary)


def main():
    st.title("Drag and Drop Image Uploader with OCR and Summarization")

    st.write("Drag and drop an image file here.")
    uploaded_file = st.file_uploader(
        "", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform OCR and summarization on the uploaded image
        perform_ocr_and_summarization(image)


if __name__ == "__main__":
    main()
