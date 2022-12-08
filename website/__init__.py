import cv2
import os
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from datetime import date
from datetime import datetime
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
# make training set 
# save faces
# face recognition
#take atendance





#veritabani yoksa oluştur 
def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
