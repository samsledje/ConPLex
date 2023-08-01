"""
Uploading and reviewing drug and target sets.
"""

import csv
import pathlib
import secrets

import flask
import fsw.views
import sqlalchemy
import werkzeug

from .. import decorators, forms, models, tasks

bp = flask.Blueprint(
    "sets",
    __name__,
    url_prefix="/sets",
)


class IndexView(fsw.views.TemplateView):
    decorators = [decorators.user_required]

    template_name = "sets/index.html.jinja"

    def get_template_context(self):
        return {
            "drug_sets": models.db_session.scalars(
                sqlalchemy.select(models.DrugSet).where(
                    models.DrugSet.user_id == flask.g.user.id
                )
            ).all(),
            "target_sets": models.db_session.scalars(
                sqlalchemy.select(models.TargetSet).where(
                    models.TargetSet.user_id == flask.g.user.id
                )
            ).all(),
        }


bp.add_url_rule("/", view_func=IndexView.as_view("index"))

class SetCreateView(fsw.views.CreateModelView):
    decorators = [decorators.user_required]

    database_session = models.db_session
    template_name = "form.html.jinja"

    # The file object from the Flask request,
    # determined by `validate_form`.
    file: werkzeug.datastructures.FileStorage

    # The number of rows within the uploaded file,
    # determined by `validate_form`.
    file_row_count: int

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

        # Check that the file is tab-separated and contains two columns,
        # and determine the number of rows within the file.
        tsv_file_reader = csv.reader(
            (bytes.decode() for bytes in self.file.stream),
            delimiter="\t",
        )

        self.file_row_count = 0
        for i, tsv_row in enumerate(tsv_file_reader):
            if len(tsv_row) != 2:
                self.request_form.file.errors.append(
                    f"The uploaded file (at row {i + 1})"
                    " does not contain two tab-separated columns."
                )
                return False

            self.file_row_count += 1

        # Reset the file stream for future use.
        self.file.stream.seek(0)

        return True

    def dispatch_valid_form_request(self):
        self.request_model_instance.user_id = flask.g.user.id
        self.request_model_instance.count = self.file_row_count

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

    def _dispatch_valid_form_request(self):
        response = super()._dispatch_valid_form_request()
        tasks.featurize_drug_set(self.request_model_instance.id)
        return response


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

    def _dispatch_valid_form_request(self):
        response = super()._dispatch_valid_form_request()
        tasks.featurize_target_set(self.request_model_instance.id)
        return response


bp.add_url_rule(
    "/target-sets/create/",
    view_func=TargetSetCreateView.as_view("create_target_set"),
    methods=["GET", "POST"],
)
