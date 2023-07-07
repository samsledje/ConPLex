import flask
import fsw.models
import sqlalchemy
import sqlalchemy.orm

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
