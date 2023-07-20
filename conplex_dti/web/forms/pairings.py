import wtforms

from .. import models
from .base import Form

BasePairingForm = Form.get_model_form(
    models.Pairing, ["drug_featurizer", "target_featurizer"]
)


class PairingForm(BasePairingForm):
    drug_set_id = wtforms.fields.SelectField("Drug Set", coerce=int)
    target_set_id = wtforms.fields.SelectField("Target Set", coerce=int)
