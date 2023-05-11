import requests
import numpy as np
import cv2

input_image_name = 'tests/test_image.jpg'
api_host = 'http://127.0.0.1:8001/'
type_rq = 'batch_inference/'

img_path = 'tests/test_image.jpg'
img_array = cv2.imread(img_path)
img_array = np.expand_dims(img_array,axis=0)
batch = np.concatenate((img_array,img_array,img_array),axis=0)
shape = batch.shape

# form a batch

data = {
    "data": batch.tolist(),
    "shape": shape
}

response = requests.post(api_host+type_rq, json=data)

data = response.json()   
print(data)