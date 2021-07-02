#gui.py

from flaskwebgui import FlaskUI
from faceRecognition.wsgi import application

FlaskUI(application).run()