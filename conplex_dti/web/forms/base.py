import flask
import fsw.forms
import wtforms


class Form(
    wtforms.Form,
    fsw.forms.CSRFProtectFormMixin,
    fsw.forms.ModelFormMixin,
):
    """
    The base form class.
    """
