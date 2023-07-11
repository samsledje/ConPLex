"""
Uploading and reviewing drug and target sets.
"""

import pathlib
import secrets

import flask
import fsw.views
import sqlalchemy

from .. import decorators, forms, models

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


class SetCreateView(fsw.views.CreateModelView):
    decorators = [decorators.user_required]

    database_session = models.db_session
    template_name = "form.html.jinja"

    file = None

    def get_redirect_url(self):
        return flask.url_for(".index")

    def validate_form(self) -> bool:
        if not super().validate_form():
            return False

        file_field_name = self.request_form.file.name
        if file_field_name not in flask.request.files:
            self.request_form.file.errors.append("No file was uploaded.")
            return False

        self.file = flask.request.files[file_field_name]
        if not self.file.filename:
            self.request_form.file.errors.append("No file was uploaded.")
            return False

        file_suffix = "".join(pathlib.Path(self.file.filename).suffixes)
        if file_suffix.lower() != ".tsv":
            self.request_form.file.errors.append(
                "The uploaded file was not a TSV file."
            )
            return False

        return True

    def dispatch_valid_form_request(self):
        self.request_model_instance.user_id = flask.g.user.id

        # Create a random (and secure) file name.
        file_suffix = "".join(pathlib.Path(self.file.filename).suffixes)
        self.file.filename = str(
            pathlib.Path(secrets.token_urlsafe(64)).with_suffix(file_suffix)
        )

        # Ensure the uploads' folder exists and save the file.
        uploads_folder_path = flask.current_app.config["UPLOADS_FOLDER_PATH"]
        uploads_folder_path.mkdir(parents=True, exist_ok=True)
        self.file.save(uploads_folder_path / self.file.filename)

        self.request_model_instance.upload_filename = self.file.filename


class DrugSetCreateView(SetCreateView):
    model = models.DrugSet
    form_class = forms.DrugSetForm

    def get_template_context(self) -> dict:
        template_context = super().get_template_context()
        template_context["title"] = "Upload a Drug Set"
        return template_context


bp.add_url_rule(
    "/drug-sets/create/",
    view_func=DrugSetCreateView.as_view("create_drug_set"),
    methods=["GET", "POST"],
)


class TargetSetCreateView(SetCreateView):
    model = models.TargetSet
    form_class = forms.TargetSetForm

    def get_template_context(self) -> dict:
        template_context = super().get_template_context()
        template_context["title"] = "Upload a Target Set"
        return template_context


bp.add_url_rule(
    "/target-sets/create/",
    view_func=TargetSetCreateView.as_view("create_target_set"),
    methods=["GET", "POST"],
)
