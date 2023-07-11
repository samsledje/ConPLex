import flask

from . import sets, users


def register_views(app: flask.Flask) -> None:
    """
    Register the application's views.
    """

    blueprints = [
        users.bp,
        sets.bp,
    ]

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    @app.route("/")
    def index():
        return flask.render_template("index.html.jinja")
