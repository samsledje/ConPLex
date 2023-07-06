import secrets

import flask


def create_app() -> flask.Flask:
    """
    Build the application.
    """
    app = flask.Flask(__name__)

    app.config.from_mapping(
        # TODO: For deployment,
        # load the key from an environment variable.
        SECRET_KEY=secrets.token_hex(64),
        # TODO: For deployment, use PostgreSQL.
        DATABASE_URL="sqlite:///web.sqlite3",
    )

    with app.app_context():
        from .models import register_models
        from .views import register_views

        register_models(app)
        register_views(app)

    return app
