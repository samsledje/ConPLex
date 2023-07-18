import flask
import huey


task_queue = huey.SqliteHuey(
    filename=flask.current_app.config["TASK_QUEUE_SQLITE_FILENAME"]
)
