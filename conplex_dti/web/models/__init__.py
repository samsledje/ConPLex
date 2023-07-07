from .base import Model as Model
from .base import db_engine as db_engine
from .base import db_session as db_session

# All models.

import flask


def register_models(app: flask.Flask) -> None:
    """
    Register the application's models.
    """

    @app.teardown_appcontext
    def remove_db_session(_=None):
        """
        Remove the database session after a request.
        See https://docs.sqlalchemy.org/en/20/orm/contextual.html.
        """
        db_session.remove()
