"""
Uploading and reviewing drug and target sets.
"""

import flask
import sqlalchemy

from .. import decorators, models

bp = flask.Blueprint(
    "sets",
    __name__,
    url_prefix="/sets",
)


@bp.route("/")
@decorators.user_required
def index():
    drug_sets = models.db_session.scalars(
        sqlalchemy.select(models.DrugSet).where(
            models.DrugSet.user_id == flask.g.user.id
        )
    ).all()
    target_sets = models.db_session.scalars(
        sqlalchemy.select(models.TargetSet).where(
            models.TargetSet.user_id == flask.g.user.id
        )
    ).all()

    return flask.render_template(
        "sets/index.html.jinja",
        drug_sets=drug_sets,
        target_sets=target_sets,
    )
