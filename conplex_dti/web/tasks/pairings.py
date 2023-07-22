import flask

from .. import models
from .base import task_queue

UPLOADS_FOLDER_PATH = flask.current_app.config["UPLOADS_FOLDER_PATH"]


@task_queue.task()
def model_pairing(pairing_id: int) -> None:
    """
    The model output will be stored within a row in the `ModelOutput` table.
    """
