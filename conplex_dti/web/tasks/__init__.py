import flask

if flask.current_app:
    from .base import task_queue
    from .sets import featurize_drug_set
else:
    from .. import create_app

    with create_app().app_context():
        from .base import task_queue
        from .sets import featurize_drug_set
