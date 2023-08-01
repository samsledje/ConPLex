import sqlalchemy.orm
import sqlalchemy.types

from .base import Model


class User(Model):
    email_address: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.types.String(64),
        doc="A valid email address for login.",
        unique=True,
    )
