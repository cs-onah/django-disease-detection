from PIL import Image
from django.http import JsonResponse

# ML Imports
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

def mobilenet_prepare_image(file):
  img = image.load_img(file, target_size = (224,224))
  img_array = image.img_to_array(img)
  # print(img_array.shape)
  img_array_expanded = np.expand_dims(img_array,axis=0)
  # print(img_array_expanded.shape)
  return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded)
  
def mobilenetV2_prepare_image(file):
  img = image.load_img(file, target_size = (224,224))
  img_array = image.img_to_array(img)
  # print(img_array.shape)
  img_array_expanded = np.expand_dims(img_array,axis=0)
  # print(img_array_expanded.shape)
  return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)
  
  
def display(classes_dict, probs):
  res = {classes_dict[i]:round(probs[i]*100,2) for i in range(len(probs)) }
  return res

def image_upload(request):
    if request.method == 'POST':
        # Assuming your form has an input field named 'image'
        uploaded_image = request.FILES.get('image')
        if uploaded_image:
            prepare_func, process_func =  mobilenetV2_prepare_image, tf.keras.applications.mobilenet_v2.preprocess_input
            model_path = "myapp/model/MobilenetV2.h5"
            model = os.path.join(model_path)
            classes_dict = {0: 'Mosaic_N', 1: 'blight_N', 2: 'brownstreak_N', 3: 'greenmite_N'}
            model = keras.models.load_model(model, compile=False)


            # Open the uploaded image using PIL
            value = Image.open(uploaded_image)
            file = value.convert('RGB')
            temp_file_path = 'image.jpg'
            file.save(temp_file_path)


            pixels = prepare_func(temp_file_path)
            pred = model.predict(pixels)
            probs = pred[0].tolist()

            response = display(classes_dict,probs)
            os.remove(temp_file_path) # delete temp file

            file.close()
            return JsonResponse(response)
        else:
            return JsonResponse({'error': 'No image uploaded.'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)