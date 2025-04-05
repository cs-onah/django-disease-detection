from PIL import Image
from django.http import JsonResponse
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO  # For handling file-like objects from request.FILES

def mobilenetV2_prepare_image(img):
    img_resized = img.resize((224, 224))  # Resize the image
    img_array = image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)

def display(classes_dict, probs):
    return {classes_dict[i]: round(probs[i] * 100, 2) for i in range(len(probs))}

def image_upload(request):
    if request.method == 'POST':
        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            prepare_func = mobilenetV2_prepare_image
            model_path = "myapp/model/MobilenetV2.h5"
            model = keras.models.load_model(model_path, compile=False)
            classes_dict = {0: 'Mosaic_N', 1: 'blight_N', 2: 'brownstreak_N', 3: 'greenmite_N'}

            # Read the image as a file-like object using PIL
            file = Image.open(uploaded_image)

            # Preprocess the image
            pixels = prepare_func(file)
            pred = model.predict(pixels)
            probs = pred[0].tolist()

            # Get prediction result
            response = display(classes_dict, probs)

            return JsonResponse(response)
        else:
            return JsonResponse({'error': 'No image uploaded.'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)