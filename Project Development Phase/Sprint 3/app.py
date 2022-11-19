from flask import Flask, Response, render_template
from camera import Video
import cv2
from keras.utils import load_img,img_to_array
import numpy as np
from skimage.transform import resize
from keras.models import load_model

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('home.html')

@app.route('/asl')
def asl():
	return render_template('asl.html')

@app.route('/predict')
def predict():
	return render_template('predict.html')

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/register')
def register():
	return render_template('register.html')

model=load_model('C:/Users/rohit/OneDrive/Desktop/VII/IBM-Project-19465-1659698319/Project Development Phase/Sprint 2/asl.h5')
def gen(video):
    while True:
        success, image = video.read()
        
        copy=image.copy()
        cv2.imwrite('img.jpg',copy)
        copy_img=load_img('img.jpg',target_size=(64,64,1))
        img=img_to_array(copy_img)
        img = resize(img,(64,64,1))
        img = np.expand_dims(img,axis=0)
        pred=np.argmax(model.predict(img))
        op=['A','B','C','D','E','F','G','H','I']
        print(op[pred])
        cv2.putText(image,op[pred],(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

video = cv2.VideoCapture(0)
@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run()

#<p class="warn"><p style="font-size:25px;text-align:center;padding-top: 150px;">Sign In to Predict</p></p>
