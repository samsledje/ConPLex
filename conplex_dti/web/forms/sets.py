import wtforms

from .. import models
from .base import Form

BaseDrugSetForm = Form.get_model_form(models.DrugSet, ["name"])
BaseTargetSetForm = Form.get_model_form(models.TargetSet, ["name"])


# NOTE: The caller must verify that the files are TSV files with valid columns.
class DrugSetForm(BaseDrugSetForm):
    file = wtforms.fields.FileField(
        "Drug TSV File",
        [wtforms.validators.InputRequired()],
        description=(
            "A TSV file with two columns,"
            " the first containing drug IDs"
            " and the second containing SMILES strings."
        ),
    )

    submit = wtforms.fields.SubmitField("Submit")


class TargetSetForm(BaseTargetSetForm):
    file = wtforms.fields.FileField(
        "Target TSV File",
        [wtforms.validators.InputRequired()],
        description=(
            "A TSV file with two columns,"
            " the first containing protein IDs"
            " and the second containing amino-acid sequences."
        ),
    )

    submit = wtforms.fields.SubmitField("Submit")
