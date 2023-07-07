import fsw.forms.models
import sqlalchemy
import wtforms

from .. import models
from .base import Form


class EmailAddressColumnFieldConverter(fsw.forms.models.StringColumnFieldConverter):
    def get_field_kwargs(self, column: sqlalchemy.Column):
        field_kwargs = super().get_field_kwargs(column)
        field_kwargs["validators"].append(wtforms.validators.Email())

        return field_kwargs


LoginForm = Form.get_model_form(
    models.User,
    ["email_address"],
    {"email_address": EmailAddressColumnFieldConverter()},
)
