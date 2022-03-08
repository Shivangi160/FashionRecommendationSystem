from flask import Flask, render_template
from flask import request
import random
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))

print(feature_list)
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend_(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=12, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

app = Flask(__name__)


# two decorators, same function
@app.route('/')
@app.route('/index.html')
def index():
    df=pd.read_csv('images.csv',usecols=['filename'])
    img_ls = [str(df.iloc[random.randint(0,44000)]["filename"]) for _ in range(12) ]
    return render_template('index.html', the_title='Home Page', img_ls=img_ls)

# @app.route('/symbol.html')
# def symbol():
#     return render_template('symbol.html', the_title='Tiger As Symbol')

@app.route('/recommend/<string:image_name>')
def recommend(image_name):
    features = feature_extraction(os.path.join("static/images",image_name),model)
    indices = recommend_(features,feature_list)
    print(len(indices))
    recommend_image_list = [filenames[indices[0][i]].split("/")[-1] for i in range(12)]
     # call the recommend scripts
    # image_name = request.args.get('image_name', default = "hello", type = str)
    return render_template('recommend.html', the_title='recommended products', image_list=recommend_image_list)

if __name__ == '__main__':
    app.run(debug=True)
