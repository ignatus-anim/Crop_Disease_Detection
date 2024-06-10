import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import tempfile
from textinfo import homeinfo, aboutinfo


# Function to load model and solution JSON based on selected crop
def load_model_and_solution(crop):
    model_file = f"models/{crop}.keras"
    solution_file = f"solutions/{crop}_solution.json"
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

# Function to display confidence
def check_confidence(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    confidence = round((float(np.max(prediction)) * 100),2)
    return confidence

# Fucntion to Attend to the Crop
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


def main():
    # SideBar
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Mode", ["Home", "About", "Disease_Prediction"])

    # HomePage
    if app_mode == "Home":
        st.header("CROP DISEASE RECOGNITION SYSTEM")
        image_path = 'photos/crops.jpeg'
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
        crop = st.selectbox('Select Crop', ['Corn','Tomato','Pepper','Apple','Banana','Cherry','Grape','Mango','Orange','Peach','Potato','Rice','Strawberry'])
        
        # Load model and solution JSON based on selected crop
        model, crop_diseases = load_model_and_solution(crop.lower())

        if model is None or crop_diseases is None:
            st.stop()

        st.write(f'You selected {crop}. Displaying {crop} disease prediction...')
        st.header(f"{crop} Disease Recognition")
        st.image(f'photos\{crop.lower()}.jpeg',use_column_width=True )
         
        # Choice between camera and file uploader
        choice = st.radio("Choose input method", ("Upload Image", "Use Camera"))

        if choice == "Upload Image":
            test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
            if test_image:
                attender(test_image=test_image, model=model, model_prediction=model_prediction, crop_diseases=crop_diseases, check_confidence=check_confidence)
        elif choice == "Use Camera":
            captured_image_path = st.camera_input("Take a picture")
            if captured_image_path:
                attender(test_image=captured_image_path, model=model, model_prediction=model_prediction, crop_diseases=crop_diseases, check_confidence=check_confidence)

# Entry point of the application
if __name__ == "__main__":
    main()
