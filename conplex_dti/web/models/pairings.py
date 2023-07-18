import sqlalchemy.orm

from .base import DrugFeaturizer, Model, TargetFeaturizer
from .sets import DrugSet, TargetSet
from .users import User


class Pairing(Model):
    user_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("User.id")
    )
    user: sqlalchemy.orm.Mapped[User] = sqlalchemy.orm.relationship()

    drug_set_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("DrugSet.id")
    )
    drug_set: sqlalchemy.orm.Mapped[DrugSet] = sqlalchemy.orm.relationship()

    target_set_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("TargetSet.id")
    )
    target_set: sqlalchemy.orm.Mapped[TargetSet] = sqlalchemy.orm.relationship()

    drug_featurizer: sqlalchemy.orm.Mapped[DrugFeaturizer]
    target_featurizer: sqlalchemy.orm.Mapped[TargetFeaturizer]
