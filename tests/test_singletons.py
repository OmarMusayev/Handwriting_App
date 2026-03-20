import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import io


def test_vocab_singleton_has_char_to_id():
    from app.core.singletons import VocabSingleton
    VocabSingleton._instance = None
    VocabSingleton.char_to_id = {}
    VocabSingleton.id_to_char = {}
    with patch("builtins.open", return_value=io.StringIO("hello\nhi\n")):
        VocabSingleton.initialize("./data/")
    assert isinstance(VocabSingleton.char_to_id, dict)
    assert " " in VocabSingleton.char_to_id
    assert len(VocabSingleton.char_to_id) > 0


def test_vocab_singleton_idx_to_char_callable():
    from app.core.singletons import VocabSingleton
    VocabSingleton._instance = None
    VocabSingleton.char_to_id = {}
    VocabSingleton.id_to_char = {}
    with patch("builtins.open", return_value=io.StringIO("hello\n")):
        VocabSingleton.initialize("./data/")
    # Get any valid id
    some_id = next(iter(VocabSingleton.id_to_char.keys()))
    ids = np.array([some_id])
    result = VocabSingleton.idx_to_char(ids)
    assert len(result) == 1


def test_vocab_singleton_only_initializes_once():
    from app.core.singletons import VocabSingleton
    VocabSingleton._instance = None
    VocabSingleton.char_to_id = {}
    VocabSingleton.id_to_char = {}
    call_count = 0
    original_open = open
    def counting_open(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return io.StringIO("ab\n")
    with patch("builtins.open", side_effect=counting_open):
        VocabSingleton.initialize("./data/")
        VocabSingleton.initialize("./data/")  # second call should be no-op
    assert call_count == 1


def test_stats_singleton_has_mean_std():
    from app.core.singletons import StatsSingleton
    StatsSingleton._initialized = False
    StatsSingleton.train_mean = None
    StatsSingleton.train_std = None
    fake_strokes = np.array(
        [np.random.randn(10, 3).astype(np.float32) for _ in range(10)],
        dtype=object
    )
    with patch("numpy.load", return_value=fake_strokes):
        StatsSingleton.initialize("./data/", n_train=8)
    assert StatsSingleton.train_mean is not None
    assert StatsSingleton.train_std is not None
    assert not np.any(StatsSingleton.train_std == 0)


def test_model_singleton_returns_same_instance():
    from app.core.singletons import ModelSingleton
    ModelSingleton._model = None
    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    with patch("app.core.singletons.HandWritingSynthesisNet", return_value=mock_model):
        with patch("torch.load", return_value={}):
            m1 = ModelSingleton.get("fake.pt", "cpu", 77)
            m2 = ModelSingleton.get("fake.pt", "cpu", 77)
    assert m1 is m2
