from keras.models import load_model
import numpy as np
import cv2
model=load_model('D:/FLASK/IMAGE/gesture.h5')
from skimage.transform import resize
def detect(frame):
        img=resize(frame,(64,64,1))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        prediction = model.predict(img)
        print(prediction)
        
        y_predict = np.argmax(model.predict(img), axis=-1)
        print(prediction)
frame=cv2.imread(r"C:\Users\VISHAL\Downloads\DATASETNEW\Dataset\image\background6.jpg")
data=detect(frame)
frame=cv2.imread(r"C:\Users\VISHAL\Downloads\DATASETNEW\Dataset\test_set\B\2.png")
data1=detect(frame)
