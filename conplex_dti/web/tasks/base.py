import flask
import huey
import torch

task_queue = huey.SqliteHuey(
    filename=flask.current_app.config["TASK_QUEUE_SQLITE_FILENAME"]
)

TORCH_DEVICE = torch.device("cuda:0")
