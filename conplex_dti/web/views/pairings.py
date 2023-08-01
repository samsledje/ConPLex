"""
Creating and analyzing data from pairings of drug and target sets.
"""

import csv

import flask
import fsw.views
import sqlalchemy

from .. import decorators, forms, helpers, models, tasks

bp = flask.Blueprint(
    "pairings",
    __name__,
    url_prefix="/pairings",
)


class IndexView(fsw.views.TemplateView):
    decorators = [decorators.user_required]

    template_name = "pairings/index.html.jinja"

    def get_template_context(self):
        return {
            "pairings": models.db_session.scalars(
                sqlalchemy.select(models.Pairing).where(
                    models.Pairing.user_id == flask.g.user.id
                )
            ).all(),
        }


bp.add_url_rule("/", view_func=IndexView.as_view("index"))


class PairingCreateView(fsw.views.CreateModelView):
    decorators = [decorators.user_required]

    database_session = models.db_session
    template_name = "form.html.jinja"

    model = models.Pairing
    form_class = forms.PairingForm

    def get_redirect_url(self):
        return flask.url_for(".index")

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

    def _dispatch_valid_form_request(self):
        response = super()._dispatch_valid_form_request()
        tasks.model_pairing(self.request_model_instance.id)
        return response


bp.add_url_rule(
    "/create/",
    view_func=PairingCreateView.as_view("create"),
    methods=["GET", "POST"],
)

UPLOADS_FOLDER_PATH = flask.current_app.config["UPLOADS_FOLDER_PATH"]


def load_drug_ids(pairing: models.Pairing) -> list[str]:
    drug_set_file_path = UPLOADS_FOLDER_PATH / pairing.drug_set.upload_filename
    with drug_set_file_path.open() as file:
        csv_reader = csv.reader(file, delimiter="\t")
        return [row[0] for row in csv_reader]


def load_target_ids(pairing: models.Pairing) -> list[str]:
    target_set_file_path = UPLOADS_FOLDER_PATH / pairing.target_set.upload_filename
    with target_set_file_path.open() as file:
        csv_reader = csv.reader(file, delimiter="\t")
        return [row[0] for row in csv_reader]


def load_model_output(pairing: models.Pairing) -> models.ModelOutput:
    return models.db_session.scalars(
        sqlalchemy.select(models.ModelOutput).where(
            models.ModelOutput.pairing_id == pairing.id
        )
    ).one()


@bp.route("<int:pairing_id>/drug-projections.json")
@decorators.user_required
def drug_projections(pairing_id: int):
    pairing = models.db_session.get(models.Pairing, pairing_id)
    if pairing.user_id != flask.g.user.id:
        flask.abort(401)

    drug_ids = load_drug_ids(pairing)

    model_output = load_model_output(pairing)
    drug_projections = helpers.convert_column_bytes_to_array(
        model_output.drug_projections
    ).reshape((-1, 2))
    drug_projections = drug_projections.tolist()

    if len(drug_projections) != len(drug_ids):
        raise RuntimeError

    return dict(zip(drug_ids, drug_projections))


@bp.route("<int:pairing_id>/target-projections.json")
@decorators.user_required
def target_projections(pairing_id: int):
    pairing = models.db_session.get(models.Pairing, pairing_id)
    if pairing.user_id != flask.g.user.id:
        flask.abort(401)

    target_ids = load_target_ids(pairing)

    model_output = load_model_output(pairing)
    target_projections = helpers.convert_column_bytes_to_array(
        model_output.target_projections
    ).reshape((-1, 2))
    target_projections = target_projections.tolist()

    if len(target_projections) != len(target_ids):
        raise RuntimeError

    return dict(zip(target_ids, target_projections))


@bp.route("<int:pairing_id>/predictions.json")
@decorators.user_required
def predictions(pairing_id: int):
    pairing = models.db_session.get(models.Pairing, pairing_id)
    if pairing.user_id != flask.g.user.id:
        flask.abort(401)

    drug_ids = load_drug_ids(pairing)
    target_ids = load_target_ids(pairing)

    model_output = load_model_output(pairing)
    predictions = helpers.convert_column_bytes_to_array(
        model_output.predictions
    ).reshape((len(drug_ids), len(target_ids)))
    predictions = predictions.tolist()

    return {
        drug_id: dict(zip(target_ids, predictions_for_drug))
        for drug_id, predictions_for_drug in zip(drug_ids, predictions)
    }


@bp.route("<int:pairing_id>/predictions.tsv")
@decorators.user_required
def predictions_tsv(pairing_id: int):
    pairing = models.db_session.get(models.Pairing, pairing_id)
    if pairing.user_id != flask.g.user.id:
        flask.abort(401)

    drug_ids = load_drug_ids(pairing)
    target_ids = load_target_ids(pairing)

    model_output = load_model_output(pairing)
    predictions = helpers.convert_column_bytes_to_array(
        model_output.predictions
    ).reshape((len(drug_ids), len(target_ids)))
    predictions = predictions.tolist()

    return "\n".join(
        f"{drug_id}\t{target_id}\t{predictions[drug_index][target_index]}"
        for drug_index, drug_id in enumerate(drug_ids)
        for target_index, target_id in enumerate(target_ids)
    ), {"Content-Type": "text/plain"}


@bp.route("<int:pairing_id>/visualization/")
@decorators.user_required
def visualization(pairing_id: int):
    pairing = models.db_session.get(models.Pairing, pairing_id)
    if pairing.user_id != flask.g.user.id:
        flask.abort(401)

    return flask.render_template(
        "pairings/visualization.html.jinja",
        pairing=pairing,
    )
