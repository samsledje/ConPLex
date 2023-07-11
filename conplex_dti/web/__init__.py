import os
import pathlib
import secrets

import flask


def create_app() -> flask.Flask:
    """
    Build the application.
    """
    app = flask.Flask(__name__)

    app.config.from_mapping(
        # Generate the secret key with `secrets.token_hex(64)`.
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        # TODO: For deployment, use PostgreSQL.
        DATABASE_URL="sqlite:///web.sqlite3",
        UPLOADS_FOLDER_PATH=pathlib.Path("web-uploads"),
        EMAIL_SMTP_SERVER_URL="smtp-mail.outlook.com",
        # Port 587 is for TLS.
        EMAIL_SMTP_PORT=587,
        # For Gmail, the SMTP username equals the sender address.
        EMAIL_SMTP_USERNAME="conplex.web@outlook.com",
        EMAIL_SMTP_PASSWORD=os.environ.get("EMAIL_SMTP_PASSWORD"),
        EMAIL_SENDER_ADDRESS="conplex.web@outlook.com",
    )

    with app.app_context():
        from .models import register_models
        from .views import register_views

        register_models(app)
        register_views(app)

    return app
