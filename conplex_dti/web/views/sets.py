"""
Uploading and reviewing drug and target sets.
"""

import flask

bp = flask.Blueprint(
    "sets",
    __name__,
    url_prefix="/sets",
)
