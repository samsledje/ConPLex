import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.types

from .base import Model
from .users import User


class Set(Model):
    __abstract__ = True

    user_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("User.id")
    )

    name: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.types.String(256)
    )

    count: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        server_default=sqlalchemy.text("0")
    )

    upload_filename: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(
        sqlalchemy.types.String(256),
        doc="The filename for the saved, uploaded TSV file.",
        unique=True,
    )

    @sqlalchemy.orm.declared_attr
    @classmethod
    def user(cls) -> sqlalchemy.orm.Mapped[User]:
        return sqlalchemy.orm.relationship(User)


class DrugSet(Set):
    pass


class TargetSet(Set):
    pass
