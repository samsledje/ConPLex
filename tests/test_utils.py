"""Tests for Utils"""
import pytest
import logging as lg
from conplex_dti import utils

@pytest.mark.parametrize(
    "test_input",
    [
        ("test1", "ProtBertFeaturizer", "MorganFeaturizer"),
        ("test2", "ESMFeaturizer", "MorganFeaturizer")
     ]
    )
def test_get_config(test_input):
    result = utils.get_config(*test_input)
    assert result.experiment_id == test_input[0]
    assert result.data.mol_feat == test_input[1]
    assert result.data.prot_feat == test_input[2]

def test_get_logger():
    result = utils.get_logger()
    assert isinstance(result, lg.Logger)

@pytest.mark.parametrize(
    "test_input",
    [
        (None, "%(asctime)s [%(levelname)s] %(message)s"),
        ("/tmp/_conplex_test_log.txt", "%(asctime)s %(message)s")
    ]
)
def test_config_logger(test_input):
    result = utils.config_logger(*test_input)
    assert isinstance(result, lg.Logger)
