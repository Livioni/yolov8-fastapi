import requests

input_image_name = 'tests/test_image.jpg'
api_host = 'http://127.0.0.1:8001/'
type_rq = 'img_object_detection_to_json'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

data = response.json()   
print(data)