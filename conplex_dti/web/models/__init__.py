import flask

# Import all models (for imports and Alembic).
from .base import BINARY_TORCH_TYPE
from .base import DrugFeaturizer as DrugFeaturizer
from .base import Model as Model
from .base import TargetFeaturizer as TargetFeaturizer
from .base import TaskStatus as TaskStatus
from .base import db_engine as db_engine
from .base import db_session as db_session
from .pairings import ModelOutput as ModelOutput
from .pairings import Pairing as Pairing
from .sets import DrugFeaturizerOutput as DrugFeaturizerOutput
from .sets import DrugSet as DrugSet
from .sets import TargetFeaturizerOutput as TargetFeaturizerOutput
from .sets import TargetSet as TargetSet
from .users import User as User


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
