import typing

import pathlib

import sqlalchemy.orm

from .base import DrugFeaturizer, Model, TargetFeaturizer, TaskStatus
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

    model_status: sqlalchemy.orm.Mapped[typing.Optional[TaskStatus]]


class ModelOutput(Model):
    pairing_id: sqlalchemy.orm.Mapped[int] = sqlalchemy.orm.mapped_column(
        sqlalchemy.ForeignKey("Pairing.id")
    )
    pairing: sqlalchemy.orm.Mapped[Pairing] = sqlalchemy.orm.relationship()

    # The two-dimensional array of ConPLex predictions,
    # the sigmoid distances between projection vectors for drugs and targets.
    # The first axis corresponds to drugs and the second to targets.
    # TODO: Expose the predictions as CSV and as JSON for visualization.
    predictions: sqlalchemy.orm.Mapped[bytes]

    # The arrays of ConPLex projection vectors for drugs and targets.
    # The projections vectors have two elements each
    # after dimensionality reduction.
    # TODO: Expose the projections as JSON for visualization.
    drug_projections: sqlalchemy.orm.Mapped[bytes]
    target_projections: sqlalchemy.orm.Mapped[bytes]


MODELS_PATH = pathlib.Path("models")
MODEL_PATHS: dict[DrugFeaturizer, dict[TargetFeaturizer, pathlib.Path]] = {
    DrugFeaturizer.MORGAN_FINGERPRINT: {
        TargetFeaturizer.PROTBERT: MODELS_PATH / "ConPLex_v1_BindingDB.pt",
    },
}
