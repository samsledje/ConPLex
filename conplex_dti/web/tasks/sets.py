import csv

import flask
import sqlalchemy
import torch

from ... import featurizer, utils
from .. import helpers, models
from .base import TORCH_DEVICE, task_queue

UPLOADS_FOLDER_PATH = flask.current_app.config["UPLOADS_FOLDER_PATH"]


@task_queue.task()
def featurize_drug_set(drug_set_id: int) -> None:
    """
    The features for all drugs in the given drug set
    will be stored as rows within the `DrugFeaturizerOutput` table.
    """
    with models.db_session() as db_session:
        drug_set = db_session.get(models.DrugSet, drug_set_id)

        drug_set.featurizer_status = models.TaskStatus.RUNNING
        db_session.commit()

        file_path = UPLOADS_FOLDER_PATH / drug_set.upload_filename
        with file_path.open() as file:
            csv_reader = csv.reader(file, delimiter="\t")
            smiles_strings = [row[1] for row in csv_reader]

        canonical_smiles_strings = set(
            utils.canonicalize(smiles_string) for smiles_string in smiles_strings
        )
        featurized_canonical_smiles_strings = set(
            db_session.scalars(
                sqlalchemy.select(models.DrugFeaturizerOutput.smiles_string).where(
                    models.DrugFeaturizerOutput.smiles_string.in_(
                        canonical_smiles_strings
                    )
                )
            )
        )
        canonical_smiles_strings_to_featurize = (
            canonical_smiles_strings - featurized_canonical_smiles_strings
        )

        if canonical_smiles_strings_to_featurize:
            # TODO: Batch.
            morgan_fingerprint_featurizer = featurizer.MorganFeaturizer().cuda(
                TORCH_DEVICE
            )
            morgan_fingerprint_outputs = [
                morgan_fingerprint_featurizer.transform(smiles_string)
                for smiles_string in canonical_smiles_strings_to_featurize
            ]

            for (
                smiles_string,
                morgan_fingerprint_output,
            ) in zip(
                canonical_smiles_strings_to_featurize,
                morgan_fingerprint_outputs,
            ):
                drug_featurizer_output = models.DrugFeaturizerOutput(
                    smiles_string=smiles_string,
                    morgan_fingerprint_output=helpers.convert_tensor_to_column_bytes(
                        morgan_fingerprint_output
                    ),
                )

                db_session.add(drug_featurizer_output)
                db_session.commit()

        drug_set.featurizer_status = models.TaskStatus.COMPLETED
        db_session.commit()


@task_queue.task()
def featurize_target_set(target_set_id: int) -> None:
    """
    The features for all targets in the given target set
    will be stored as rows within the `TargetFeaturizerOutput` table.
    """
    with models.db_session() as db_session:
        target_set = db_session.get(models.TargetSet, target_set_id)

        target_set.featurizer_status = models.TaskStatus.RUNNING
        db_session.commit()

        file_path = UPLOADS_FOLDER_PATH / target_set.upload_filename
        with file_path.open() as file:
            csv_reader = csv.reader(file, delimiter="\t")
            sequences = [row[1] for row in csv_reader]

        sequences = set(sequences)
        featurized_sequences = set(
            db_session.scalars(
                sqlalchemy.select(models.TargetFeaturizerOutput.sequence).where(
                    models.TargetFeaturizerOutput.sequence.in_(sequences)
                )
            )
        )
        sequences_to_featurize = sequences - featurized_sequences

        if sequences_to_featurize:
            # TODO: Batch.
            protbert_featurizer = featurizer.ProtBertFeaturizer().cuda(TORCH_DEVICE)
            protbert_outputs = [
                protbert_featurizer.transform(sequence)
                for sequence in sequences_to_featurize
            ]

            for (
                sequence,
                protbert_output,
            ) in zip(
                sequences_to_featurize,
                protbert_outputs,
            ):
                target_featurizer_output = models.TargetFeaturizerOutput(
                    sequence=sequence,
                    protbert_output=helpers.convert_tensor_to_column_bytes(
                        protbert_output
                    ),
                )

                db_session.add(target_featurizer_output)
                db_session.commit()

        target_set.featurizer_status = models.TaskStatus.COMPLETED
        db_session.commit()
