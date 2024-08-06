import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

def init_data_set():
    rf = Roboflow(api_key=os.getenv('API_KEY'))

    project = rf.workspace("tesis-qydta").project("planos-craneales-prenatales")
    dataset = project.version(3).download("yolov8")

    return dataset
