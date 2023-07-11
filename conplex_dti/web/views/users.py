"""
User authentication and sessions.

Authentication is performed by sending users login links.
User sessions are managed by setting `user_id` in `flask.session`.
If the user ID exists, `flask.g.user` is set to the `User` instance
corresponding to the current user.
"""

import os
import secrets
import smtplib
import ssl
import threading

import flask
import fsw.views
import sqlalchemy

from .. import forms, models

bp = flask.Blueprint(
    "users",
    __name__,
    url_prefix="/users",
)

LOGIN_TOKEN_TO_USER_MAP: dict[str, models.User] = {}


@bp.before_app_request
def load_user() -> None:
    if "user_id" in flask.session:
        flask.g.user = models.db_session.get(models.User, flask.session["user_id"])


class LoginView(fsw.views.FormView):
    template_name = "form.html.jinja"
    form_class = forms.LoginForm

    def get_redirect_url(self):
        return flask.url_for(".login_confirmation")

    def get_template_context(self):
        template_context = super().get_template_context()
        template_context["title"] = "Log In"
        return template_context

    def dispatch_valid_form_request(self) -> None:
        user = models.db_session.scalars(
            sqlalchemy.select(models.User).where(
                models.User.email_address == self.request_form.email_address.data
            )
        ).first()

        if user is None:
            user = models.User(email_address=self.request_form.email_address.data)
            user.save()

        flask.session["login_email_address"] = user.email_address

        # Generate a login token for the user.
        login_token = secrets.token_urlsafe(64)
        LOGIN_TOKEN_TO_USER_MAP[login_token] = user

        # Invalidate the token after 5 minutes.
        threading.Timer(
            300,
            lambda: LOGIN_TOKEN_TO_USER_MAP.pop(login_token, None),
        ).start()

        # Send an email to the user with a login link (see Real Python).
        ssl_context = ssl.create_default_context()
        with smtplib.SMTP(
            flask.current_app.config["EMAIL_SMTP_SERVER_URL"],
            flask.current_app.config["EMAIL_SMTP_PORT"],
        ) as smtp_server:
            smtp_server.starttls(context=ssl_context)
            smtp_server.login(
                flask.current_app.config["EMAIL_SMTP_USERNAME"],
                flask.current_app.config["EMAIL_SMTP_PASSWORD"],
            )
            smtp_server.sendmail(
                flask.current_app.config["EMAIL_SENDER_ADDRESS"],
                user.email_address,
                (
                    "Subject: ConPLex Web Login Link"
                    "\n"
                    "\nHello,"
                    "\n"
                    "\nYour login link for ConPLex Web is:"
                    "\n"
                    f"\n{flask.url_for('users.login_with_token', token=login_token, _external=True)}"
                    "\n"
                    "\nThis link will become invalid after five minutes."
                ),
            )


bp.add_url_rule(
    "/login/",
    view_func=LoginView.as_view("login"),
    methods=["GET", "POST"],
)


class LoginConfirmationView(fsw.views.TemplateView):
    template_name = "users/login-confirmation.html.jinja"

    def get_template_context(self) -> dict:
        template_context = super().get_template_context()

        if "login_email_address" not in flask.session:
            flask.abort(404)

        template_context["login_email_address"] = flask.session["login_email_address"]

        return template_context


bp.add_url_rule(
    "/login-confirmation/",
    view_func=LoginConfirmationView.as_view("login_confirmation"),
)


@bp.route("/login/<token>/")
def login_with_token(token: str):
    if token not in LOGIN_TOKEN_TO_USER_MAP:
        flask.abort(404)

    flask.session["user_id"] = LOGIN_TOKEN_TO_USER_MAP.pop(token).id

    return flask.redirect(flask.url_for("index"))


@bp.route("/logout/")
def logout():
    flask.session.pop("user_id", None)

    return flask.redirect(flask.url_for("index"))
