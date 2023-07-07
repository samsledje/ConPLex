import flask


def register_views(app: flask.Flask) -> None:
    """
    Register the application's views.
    """

    blueprints = []

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

    @app.route("/")
    def index():
        return "Index"
