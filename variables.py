import os
target_size = (224, 224)
thres_neighbours = 5
seed = 1234
host = '0.0.0.0'
port = 5000
found_table = 'found_dog'
lost_table = 'lost_dog'
root_password = '1234'
db_url = 'mysql+pymysql://root:{}@localhost:3306/doggy_similarity'.format(root_password)
local_url = 'http://0.0.0.0:5000/predict'

# data directories and model paths
test_dir = 'data/Test_Images/'
test_data_path = 'data/weights/Test_data.npz'
model_converter = "data/weights/model.tflite"
n_neighbour_weights = 'data/weights/nearest_neighbour.pkl'
found_folder_id = "1NXssdxcNBrW6-B5qr8RGx2F2rZXX9XIk"
lost_folder_id = "1yZQciBpWKMP2SisH7zp_25AWJUPM1aJD"
test_folder_id = "1ZB25Xz4GVjhgXuO3Sqcw3E1viQKmCSAb"
client_id = "236819335465-43d7n42tsg7ompebj9cf7cb3c5vg5lvj.apps.googleusercontent.com"
client_secret = "JxTRsQIpvPZIhoJvHr_kKDCJ"

cloud_image_dir = "https://res.cloudinary.com/douc1omvg/image/upload/Test_Images/"