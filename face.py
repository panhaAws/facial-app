import streamlit as st
import pickle
import cv2
from skimage import io
from skimage.transform import resize
from skimage.feature import hog
from PIL import Image
from skimage import feature, transform
import lzma
import base64
from io import BytesIO
import numpy as np
import os

# Get the absolute path of the current script
current_script_path = os.path.realpath(__file__)

# Define the relative path to your model file
relative_model_path = "svm_model_hog_64_128_c10_poly_60.pkl"

# Construct the absolute path to the model file
absolute_model_path = os.path.join(os.path.dirname(current_script_path), relative_model_path)
# Load the trained SVM model
with open(absolute_model_path, 'rb') as model_file:
    clf = pickle.load(model_file)

class_labels = clf.classes_

import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Function to perform HOG feature extraction on an image
def extract_hog_features2(image):
    resized_img = resize(image, (64, 64))  # Assuming your model was trained on 64x64 images
    hog_features = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), multichannel=True)
    return hog_features


def detect_and_crop_faces_in_image(input_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to 8-bit
    image = cv2.convertScaleAbs(input_image)

    # Check if the image is already grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, crop it out
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = image[y:y + h, x:x + w]
        return cropped_face
    return image


def normalize_images(image):
    # Normalize the images and convert them to 8-bit

    image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
    image = cv2.convertScaleAbs(image)

    return image


def extract_hog_features(image):
    image_resized = transform.resize(image, (128, 64))  # height 128, width 64

    # Check if the image is grayscale or color
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        # The image is a color image, convert it to grayscale
        image_resized_32 = image_resized.astype(np.float32)
        gray = cv2.cvtColor(image_resized_32, cv2.COLOR_BGR2GRAY)
    else:
        # The image is already grayscale
        gray = image_resized

    # Extract HOG features
    hog_features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                               block_norm='L2-Hys')
    return hog_features


# Streamlit app
def main():
    st.title("Facial Recognition App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        image = io.imread(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300, use_column_width=False)

        if st.button('Show result'):
            # resize to 250*250
            image = Image.fromarray(image).resize((250, 250))

            # Convert to gray scale
            image = image.convert(mode='L')

            image = np.array(image)
            # normalize
            # image = image/255

            # normalize again base on daro step
            image = normalize_images(image)

            # Face detection
            image = detect_and_crop_faces_in_image(image)

            # Display the uploaded image

            # st.write(image)
            # Extract HOG features
            hog_features = extract_hog_features(image)

            # Make prediction using the trained SVM model
            prediction = clf.predict([hog_features])[0]
            probability_estimates = clf.predict_proba([hog_features])
            prob_label_pairs = list(zip(probability_estimates[0], class_labels))

            # Sort based on probabilities (in descending order)
            sorted_pairs = sorted(prob_label_pairs, key=lambda x: x[0], reverse=True)

            # Display the person with the highest probability
            top_person_prob, top_person_label = sorted_pairs[0]
            top_person_prob_path = os.path.join(os.path.dirname(current_script_path),
                                                f"test/{top_person_label}/{top_person_label}_0001.jpg")
            # Check if the image file exists
            if os.path.isfile(top_person_prob_path):
                # Use cv2.imread to load the image
                image = cv2.imread(top_person_prob_path)
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Display the image
                with st.container():
                    st.image(image, width=250, use_column_width=False)
                    st.write(f"{top_person_label} : {top_person_prob * 100:.2f}%")

                # Create columns for the next 5 persons
                col1, col2, col3, col4, col5 = st.columns(5)

                # Display images and probabilities of the next 5 persons
                for i in range(1, min(6, len(sorted_pairs))):
                    prob, label = sorted_pairs[i]
                    person_path = os.path.join(os.path.dirname(current_script_path),
                                               f"test/{label}/{label}_0001.jpg")

                    # Check if the image file exists
                    if os.path.isfile(person_path):
                        # Use cv2.imread to load the image
                        image = cv2.imread(person_path)
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Display the image with name and probability below in the respective column
                        if i == 1:
                            with col1:
                                st.image(image, width=120, use_column_width=False)
                                st.markdown(f'<p style="font-size: 10px;">{label} : {prob * 100:.2f}%</p>',
                                            unsafe_allow_html=True)
                        elif i == 2:
                            with col2:
                                st.image(image, width=120, use_column_width=False)
                                st.markdown(f'<p style="font-size: 10px;">{label} : {prob * 100:.2f}%</p>',
                                            unsafe_allow_html=True)
                        elif i == 3:
                            with col3:
                                st.image(image, width=120, use_column_width=False)
                                st.markdown(f'<p style="font-size: 10px;">{label} : {prob * 100:.2f}%</p>',
                                            unsafe_allow_html=True)
                        elif i == 4:
                            with col4:
                                st.image(image, width=120, use_column_width=False)
                                st.markdown(f'<p style="font-size: 10px;">{label} : {prob * 100:.2f}%</p>',
                                            unsafe_allow_html=True)
                        elif i == 5:
                            with col5:
                                st.image(image, width=120, use_column_width=False)
                                st.markdown(f'<p style="font-size: 10px;">{label} : {prob * 100:.2f}%</p>',
                                            unsafe_allow_html=True)

                    else:
                        st.error(f"Image not found at path: {person_path}")


if __name__ == '__main__':
    main()
