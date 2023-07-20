"""
Creating and analyzing data from pairings of drug and target sets.
"""

import flask
import fsw.views
import sqlalchemy

from .. import decorators, forms, models

bp = flask.Blueprint(
    "pairings",
    __name__,
    url_prefix="/pairings",
)


class PairingCreateView(fsw.views.CreateModelView):
    decorators = [decorators.user_required]

    database_session = models.db_session
    template_name = "form.html.jinja"

    model = models.Pairing
    form_class = forms.PairingForm

    def get_redirect_url(self):
        # TODO: Create a `pairings.index` view.
        return flask.url_for("index")

    def get_template_context(self) -> dict:
        template_context = super().get_template_context()
        template_context["title"] = "Create a Pairing"
        return template_context

    def get_form(self):
        form = super().get_form()

        form.drug_set_id.choices = [
            (drug_set.id, drug_set.name)
            for drug_set in models.db_session.scalars(
                sqlalchemy.select(models.DrugSet)
                .where(models.DrugSet.user_id == flask.g.user.id)
                .where(models.DrugSet.featurizer_status == models.TaskStatus.COMPLETED)
            )
        ]
        form.target_set_id.choices = [
            (target_set.id, target_set.name)
            for target_set in models.db_session.scalars(
                sqlalchemy.select(models.TargetSet)
                .where(models.TargetSet.user_id == flask.g.user.id)
                .where(
                    models.TargetSet.featurizer_status == models.TaskStatus.COMPLETED
                )
            )
        ]

        return form

    def dispatch_valid_form_request(self):
        self.request_model_instance.user_id = flask.g.user.id


bp.add_url_rule(
    "/create/",
    view_func=PairingCreateView.as_view("create"),
    methods=["GET", "POST"],
)
