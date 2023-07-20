"""
Creating and analyzing data from pairings of drug and target sets.
"""

import flask

bp = flask.Blueprint(
    "pairings",
    __name__,
    url_prefix="/pairings",
)
