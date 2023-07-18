import datetime
import typing

import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.types

from .base import Model
from .users import User


# When a drug or target set expires,
# its CSV file is deleted, while the preprocessed features are retained.
# TODO: Create a Huey task to delete expired CSV files each day.
# TODO: Note that stored times are in UTC.
EXPIRATION_TIMEDELTA = datetime.timedelta(weeks=2)


class Set(Model):
    __abstract__ = True

    user_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("User.id")
    )

    name: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.types.String(256)
    )

    count: sqlalchemy.orm.Mapped[typing.Optional[int]]

    upload_filename: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.types.String(256),
        doc="The filename for the saved, uploaded TSV file.",
        unique=True,
    )

    def get_expired_at(self) -> datetime.datetime:
        return self.created_at + EXPIRATION_TIMEDELTA

    @sqlalchemy.orm.declared_attr
    @classmethod
    def user(cls) -> sqlalchemy.orm.Mapped[User]:
        return sqlalchemy.orm.relationship(User)


class DrugSet(Set):
    pass


class TargetSet(Set):
    pass
