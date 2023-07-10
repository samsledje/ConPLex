import flask

from . import users


def register_views(app: flask.Flask) -> None:
    """
    Register the application's views.
    """

    blueprints = [
        users.bp,
    ]

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    @app.route("/")
    def index():
        return "Index"
