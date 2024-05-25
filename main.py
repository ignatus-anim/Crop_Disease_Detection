import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import cv2
import tempfile
from textinfo import homeinfo, aboutinfo


# Attend to the Crop

def attender(test_image, model, crop_diseases, model_prediction, check_confidence):
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.write("Please upload an image first")

    if st.button("Predict"):
        if test_image is not None:
            result_index = model_prediction(test_image, model)
            confidence = check_confidence(test_image, model)
            disease_name = crop_diseases[result_index]["name"]
            st.success(f"Model is Predicting, It is a {disease_name}")
            st.success(f"Confidence = {confidence}%")
        else:
            st.write("Please upload an image first")

    if st.button("Show Cause"):
        if test_image is not None:
            result_index = model_prediction(test_image, model)
            disease_cause = crop_diseases[result_index]["causes"]

            st.write("Causes include:")
            for cause in disease_cause:
                st.success(cause)
        else:
            st.write("Please upload an image first")

    if st.button("Recommend Solution"):
        if test_image is not None:
            result_index = model_prediction(test_image, model)
            disease_solution = crop_diseases[result_index]["recommended_solutions"]

            st.write("Some Recommended solutions include:")
            for solution in disease_solution:
                st.success(solution)

    if st.button("Recommend Pesticide"):
        if test_image is not None:
            result_index = model_prediction(test_image, model)
            disease_pesticide = crop_diseases[result_index]["recommended_pesticide"]

            st.write("Some Recommended solutions include:")
            for pesticide in disease_pesticide:
                st.success(pesticide)

    # else:
    #     st.write("Please upload an image first")

    #     use_camera = st.checkbox("Use Camera")
    #     if use_camera:
    #         captured_image_path = capture_image_from_camera()
    #         if captured_image_path is not None:
    #             st.image(captured_image_path, use_column_width=True)

    #             if st.button("Predict from Camera"):
    #                 result_index = model_prediction(captured_image_path, model)
    #                 disease_name = crop_diseases[result_index]["name"]
    #                 st.success(f"Model is Predicting, It is a {disease_name}")

    #             if st.button("Show Cause from Camera"):
    #                 result_index = model_prediction(captured_image_path, model)
    #                 disease_cause = crop_diseases[result_index]["causes"]

    #                 st.write("Causes include:")
    #                 for cause in disease_cause:
    #                     st.success(cause)

    #             if st.button("Recommend Solution from Camera"):
    #                 result_index = model_prediction(captured_image_path, model)
    #                 disease_solution = crop_diseases[result_index]["recommended_solutions"]

    #                 st.write("Some Recommended solutions include:")
    #                 for solution in disease_solution:
    #                     st.success(solution)


# Function to load model and solution JSON based on selected crop
def load_model_and_solution(crop):
    model_file = f"{crop}.keras"
    solution_file = f"{crop}_solution.json"
    model = tf.keras.models.load_model(model_file)
    with open(solution_file, 'r') as solutions:
        crop_diseases = json.load(solutions)
    return model, crop_diseases


# Function to predict disease based on selected crop
def model_prediction(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Convert single image to a batch
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


def check_confidence(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    confidence = round(float(np.max(prediction)) * 100)
    return confidence


# Function to capture image from camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.write("Unable to open the camera")
        return None

    frame_window = st.image([])
    captured_image = None

    capture_button = st.button("Capture Image")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

        if capture_button:
            captured_image = frame
            cap.release()
            cv2.destroyAllWindows()
            break

    if captured_image is not None:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        # Save the captured image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            captured_image = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_file.name, captured_image)
            temp_file.seek(0)
            st.success("Image captured successfully")
            return temp_file.name
    return None


def main():
    # SideBar
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Crop", ["Home", "About", "Disease_Prediction"])

    # HomePage
    if app_mode == "Home":
        st.header("CROP DISEASE RECOGNITION SYSTEM")
        image_path = 'home.jpeg'
        st.image(image_path, use_column_width=True)
        st.markdown(homeinfo)

    # About Page
    elif app_mode == "About":
        st.header("About")
        st.markdown(aboutinfo)

    # Disease Prediction Page
    elif app_mode == 'Disease_Prediction':
        st.title('Disease Prediction')
        st.write('Select a crop to predict diseases.')

        # Crop selection
        crop = st.selectbox('Select Crop', ['Corn', 'Pepper', 'Tomato'])
        # Load model and solution JSON based on selected crop
        model, crop_diseases = load_model_and_solution(crop.lower())
        if crop == 'Corn':
            st.write('You selected Corn. Displaying Corn disease prediction...')
            st.header(f"{crop} Disease Recognition")
            test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
            attender(test_image=test_image, model=model, model_prediction=model_prediction, crop_diseases=crop_diseases,
                     check_confidence=check_confidence)
            # Add your disease prediction logic for Corn here
        elif crop == 'Pepper':
            st.write('You selected Pepper. Displaying Pepper disease prediction...')
            st.header(f"{crop} Disease Recognition")
            test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
            attender(test_image=test_image, model=model, model_prediction=model_prediction, crop_diseases=crop_diseases,
                     check_confidence=check_confidence)
            # Add your disease prediction logic for Pepper here
        elif crop == 'Tomato':
            st.write('You selected Tomato. Displaying Tomato disease prediction...')
            st.header(f"{crop} Disease Recognition")
            test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
            attender(test_image=test_image, model=model, model_prediction=model_prediction, crop_diseases=crop_diseases,
                     check_confidence=check_confidence)
            # Add your disease prediction logic for Tomato here


# Entry point of the application
if __name__ == "__main__":
    main()
