import csv

import flask
import sklearn.manifold
import sqlalchemy
import torch

from ... import utils
from ...model import architectures
from .. import helpers, models
from ..models.pairings import MODEL_PATHS
from ..models.sets import (
    DRUG_FEATURIZER_TO_OUTPUT_COLUMN_NAME,
    TARGET_FEATURIZER_TO_OUTPUT_COLUMN_NAME,
)
from .base import TORCH_DEVICE, task_queue

UPLOADS_FOLDER_PATH = flask.current_app.config["UPLOADS_FOLDER_PATH"]


@task_queue.task()
def model_pairing(pairing_id: int) -> None:
    """
    The model output will be stored within a row in the `ModelOutput` table.
    """
    with models.db_session() as db_session:
        pairing = db_session.get(models.Pairing, pairing_id)

        pairing.model_status = models.TaskStatus.RUNNING
        db_session.commit()

        drug_set_file_path = UPLOADS_FOLDER_PATH / pairing.drug_set.upload_filename
        target_set_file_path = UPLOADS_FOLDER_PATH / pairing.target_set.upload_filename

        with drug_set_file_path.open() as file:
            csv_reader = csv.reader(file, delimiter="\t")
            smiles_strings = [row[1] for row in csv_reader]
        canonical_smiles_strings = [
            utils.canonicalize(smiles_string) for smiles_string in smiles_strings
        ]
        drug_featurizer_output_column = getattr(
            models.DrugFeaturizerOutput,
            DRUG_FEATURIZER_TO_OUTPUT_COLUMN_NAME[pairing.drug_featurizer],
        )
        drug_set_feature_bytes = [
            db_session.scalars(
                sqlalchemy.select(drug_featurizer_output_column).where(
                    models.DrugFeaturizerOutput.smiles_string == smiles_string
                )
            ).one()
            for smiles_string in canonical_smiles_strings
        ]
        drug_set_features = [
            helpers.convert_column_bytes_to_tensor(feature_bytes).to(TORCH_DEVICE)
            for feature_bytes in drug_set_feature_bytes
        ]

        with target_set_file_path.open() as file:
            csv_reader = csv.reader(file, delimiter="\t")
            sequences = [row[1] for row in csv_reader]
        target_featurizer_output_column = getattr(
            models.TargetFeaturizerOutput,
            TARGET_FEATURIZER_TO_OUTPUT_COLUMN_NAME[pairing.target_featurizer],
        )
        target_set_feature_bytes = [
            db_session.scalars(
                sqlalchemy.select(target_featurizer_output_column).where(
                    models.TargetFeaturizerOutput.sequence == sequence
                )
            ).one()
            for sequence in sequences
        ]
        target_set_features = [
            helpers.convert_column_bytes_to_tensor(feature_bytes).to(TORCH_DEVICE)
            for feature_bytes in target_set_feature_bytes
        ]

        model = architectures.SimpleCoembeddingNoSigmoid()
        model.load_state_dict(
            torch.load(
                MODEL_PATHS[pairing.drug_featurizer][pairing.target_featurizer],
                map_location=TORCH_DEVICE,
            )
        )
        model = model.eval()
        model = model.to(TORCH_DEVICE)

        # TODO: Batch.
        drug_projections = [
            model.drug_projector(torch.stack([drug_feature_vector]))
            for drug_feature_vector in drug_set_features
        ]
        target_projections = [
            model.target_projector(torch.stack([target_feature_vector]))
            for target_feature_vector in target_set_features
        ]

        t_sne = sklearn.manifold.TSNE(2, perplexity=5)
        projections_under_t_sne = t_sne.fit_transform(
            torch.concatenate(
                [torch.stack(drug_projections), torch.stack(target_projections)]
            )
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # TODO: Sigmoid.
        model_output = models.ModelOutput(
            pairing_id=pairing.id,
            predictions=helpers.convert_tensor_to_column_bytes(
                torch.Tensor(
                    [
                        [
                            model.activator(drug_projection, target_projection)
                            for target_projection in target_projections
                        ]
                        for drug_projection in drug_projections
                    ]
                )
            ),
            drug_projections=helpers.convert_array_to_column_bytes(
                projections_under_t_sne[: len(drug_projections)]
            ),
            target_projections=helpers.convert_array_to_column_bytes(
                projections_under_t_sne[-len(target_projections) :]
            ),
        )

        db_session.add(model_output)
        db_session.commit()
