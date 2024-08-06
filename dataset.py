from roboflow import Roboflow


def init_data_set():
    rf = Roboflow(api_key="0XhrECkopF3lftCbfGV2")

    project = rf.workspace("tesis-qydta").project("planos-craneales-prenatales")
    dataset = project.version(3).download("yolov8")

    return dataset
