from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
flask_app = Flask(__name__)
flask_app.config.from_object(Config)
db = SQLAlchemy(flask_app)
from app import routes,models
