import joblib
from flask import Flask, request, render_template
import base64
import cv2
import numpy as np


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/',methods=['POST'])
def predict():
    image= request.form['data']
    img = np.frombuffer(base64.b64decode(image[22:]), dtype=np.uint8)
    character = cv2.imdecode(img, 0)
    resized_character = cv2.resize(character, (28, 28)).astype('int')  
    processed = resized_character.reshape((1, 784))
    processed = processed
    # img = Image.open('./static/image.png')
    # numpydata = asarray(img)
    # im = numpydata[:,:,0].reshape(1,784)
    # im = im_28.reshape(1, 784)
    model = joblib.load(open('model_random_forest.pkl', 'rb'))
    pred = model.predict(processed)
    return str(pred[0])


# img = Image.open('test.png')
# numpydata = asarray(img)
# data2 = numpydata[:,:,0].reshape(1,784)
# data = data2.reshape(28,28)
# print(data2)
# file = open('model_random_forest.pkl','rb')
# model = pickle.load(file)
# out = model.predict(data2)
# print(out)


# im = cv2.imread('test.png')
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = cv2.GaussianBlur(im, (15, 15), 0)
# ret,im_th = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
# im_28 = cv2.resize(im, (28, 28),interpolation=cv2.INTER_AREA)
# im = im_28.reshape(1, 784)
# plt.axis('off')
# plt.imshow(im_28,cmap=matplotlib.cm.binary,
#            interpolation='nearest')
# plt.show()
# file = open('model_random_forest.pkl','rb')
# model = pickle.load(file)
# out = model.predict(im)
# print(out)


if __name__ == "__main__":
    app.run()