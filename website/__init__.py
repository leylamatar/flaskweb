import cv2
import os
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from datetime import date
import datetime
from flask import Flask,request,render_template,Response
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

db = SQLAlchemy()
DB_NAME = "database.db"

# creat flask application


def create_app():
    app = Flask(__name__)
    # uygulama için koruma kod lazım
    app.config['SECRET_KEY'] = 'heeellslslslsl kdkdkdk'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth
# how to access the url
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User

    with app.app_context():
        db.create_all()

        
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    return app

#### Saving Date today in 2 different formats
def datetoday():
    return date.today().strftime("%m_%d_%y")
def datetoday2():
    return date.today().strftime("%d-%B-%Y")

#open cam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#create attendance file
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    x = datetime.date.today()
    with open(f'Attendance//Attendance_{x.day}-{x.month}-{x.year}.csv','w') as f:
        f.write('Name,school ID,Time')


# extract the face from an image to make a training set
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #make the face gray
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# Identify face using ML(machine learning) model 

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# make training set 
# save faces
# face recognition
#take atendance





#veritabani yoksa oluştur 
def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
