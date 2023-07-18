from .base import task_queue


@task_queue.task()
def featurize_drug_set(drug_set_id: int) -> None:
    """
    The features for all drugs in the given drug set
    will be stored as rows within the `DrugFeaturizerOutput` table.
    """


@task_queue.task()
def featurize_target_set(target_set_id: int) -> None:
    """
    The features for all targets in the given target set
    will be stored as rows within the `TargetFeaturizerOutput` table.
    """
