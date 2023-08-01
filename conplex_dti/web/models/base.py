import enum

import flask
import fsw.models
import numpy as np
import sqlalchemy
import sqlalchemy.orm
import torch

db_engine = sqlalchemy.create_engine(flask.current_app.config["DATABASE_URL"])
db_session = sqlalchemy.orm.scoped_session(sqlalchemy.orm.sessionmaker(bind=db_engine))


class Model(
    sqlalchemy.orm.DeclarativeBase,
    fsw.models.ClassNameModelMixin,
    fsw.models.IDModelMixin,
    fsw.models.SaveModelMixin,
    fsw.models.CreateTimestampModelMixin,
    fsw.models.UpdateTimestampModelMixin,
    fsw.models.DeleteTimestampModelMixin,
):
    """
    The base model class.
    """

    __abstract__ = True

    # The database session for FSW.
    database_session = db_session


class TaskStatus(enum.Enum):
    """
    The status of an executed Huey task.
    Some models will include columns of the form
    `task_status: sqlalchemy.orm.Mapped[typing.Optional[TaskStatus]]`.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DrugFeaturizer(enum.Enum):
    MORGAN_FINGERPRINT = "Morgan Fingerprint"


class TargetFeaturizer(enum.Enum):
    PROTBERT = "ProtBert"


BINARY_TORCH_TYPE = torch.float32
BINARY_NUMPY_TYPE = np.float32
