import os
import json
import numpy as np
import cv2 as cv
import requests
from PIL import Image
from util import *
from flask import Flask
from flask import jsonify
from flask import request

from variables import *
from heroku_inference import predict_neighbour, extract_features

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
drive = GoogleDrive(gauth)
neighbor = extract_features()

app = Flask(__name__)

def preprocess_image(image):
    if image.shape[-1] == 1:
        return False
    else:
        image = cv.resize(image, target_size, cv.INTER_AREA)
        return image

def get_image_path(folder_id):
    img_arr = drive.ListFile(
                    {'q': "'{}' in parents and trashed=false".format(folder_id)}
                    ).GetList()
    if len(img_arr) > 0:
        img_idx = max([(int(img_path['title'].split('.')[0])) for img_path in img_arr]) + 1
        img_name = str(img_idx)+'.png'
    else:
        img_name = '1.png'
    return img_name
    
def update_found_table(img_path):
    engine = create_engine(db_url)
    if table_name in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_sql_table(table_name, db_url)
        df_length = len(data.values)
        data.loc[df_length+1] = img_path
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        print("Create a Table named {}".format(table_name))

def upload_to_gdrive(img_path, upload_image, folder_id):
    cv.imwrite(img_path, upload_image)
    gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
    gfile.SetContentFile(img_path)

    gfile.Upload()
    os.remove(img_path)


@app.route("/found", methods=["POST"])
def found():
    dogimagefile= request.files['image'].read()
    dogimage = np.fromstring(dogimagefile, np.uint8)
    dogimage = cv.imdecode(dogimage,cv.IMREAD_COLOR) 
    
    processed_image = preprocess_image(dogimage)
    img_path = get_image_path(found_folder_id)
    upload_to_gdrive(img_path, processed_image, found_folder_id)
    # update_found_table(img_path)

    response = {
        'uploaded_image': 'Successfully Uploaded the image as {}'.format(img_path)
    }
    return jsonify(response)

@app.route("/lost", methods=["POST"])
def predict(neighbor=neighbor):

      dogimagefile= request.files['image'].read()
      dogimage = np.fromstring(dogimagefile, np.uint8)
      dogimage = cv.imdecode(dogimage,cv.IMREAD_COLOR) 
      dogimage = preprocess_image(dogimage)
      if dogimage.any():
        img_path = get_image_path(lost_folder_id)
        upload_to_gdrive(img_path, dogimage, lost_folder_id)   
        dogimage = preprocessing_function(dogimage)
        dogimage = dogimage.astype(np.float32)
        n_neighbours = predict_neighbour(dogimage, img_path, neighbor)

        response = {
                    "n_neighbours": n_neighbours
                   }
        return jsonify(response)

      else:
        return "Please Insert RGB image of your DOG !"


if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=False)