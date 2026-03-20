# app/core/singletons.py
import numpy as np
import torch
from collections import Counter
from models.models import HandWritingSynthesisNet
from utils.data_utils import train_offset_normalization


class VocabSingleton:
    _instance = None
    char_to_id: dict = {}
    id_to_char: dict = {}

    @classmethod
    def initialize(cls, data_path: str):
        if cls._instance is not None:
            return
        with open(data_path + "sentences.txt") as f:
            texts = f.read().splitlines()
        char_seqs = [list(t) for t in texts]
        max_len = max(len(s) for s in char_seqs)
        padded = [s + [" "] * (max_len - len(s)) for s in char_seqs]
        counter = Counter()
        for seq in padded:
            counter.update(seq)
        unique = sorted(counter)
        cls.id_to_char = dict(enumerate(unique))
        cls.char_to_id = {v: k for k, v in cls.id_to_char.items()}
        cls._instance = True

    @classmethod
    def idx_to_char(cls, id_seq) -> np.ndarray:
        return np.array([cls.id_to_char[int(i)] for i in id_seq])

    @classmethod
    def vocab_size(cls) -> int:
        return len(cls.id_to_char)


class StatsSingleton:
    _initialized = False
    train_mean: np.ndarray = None
    train_std: np.ndarray = None

    @classmethod
    def initialize(cls, data_path: str, n_train: int = None):
        if cls._initialized:
            return
        strokes = np.load(data_path + "strokes.npy", allow_pickle=True, encoding="bytes")
        lengths = [len(s) for s in strokes]
        max_len = max(lengths)
        n_total = len(strokes)
        if n_train is None:
            n_train = int(0.9 * n_total)
        data = np.zeros((n_total, max_len, 3), dtype=np.float32)
        for i, (s, l) in enumerate(zip(strokes, lengths)):
            data[i, :l] = s
        train_data = data[:n_train]
        mean = train_data[:, :, 1:].mean(axis=(0, 1))
        std = train_data[:, :, 1:].std(axis=(0, 1))
        std = np.where(std == 0, 1.0, std)
        cls.train_mean = mean
        cls.train_std = std
        cls._initialized = True
        del strokes, data, train_data  # free RAM immediately


class ModelSingleton:
    _model = None

    @classmethod
    def get(cls, model_path: str, device: str, vocab_size: int):
        if cls._model is None:
            model = HandWritingSynthesisNet(window_size=vocab_size)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            cls._model = model
        return cls._model


def startup_singletons(data_path: str, model_path: str, device: str):
    """Call once at app startup."""
    VocabSingleton.initialize(data_path)
    StatsSingleton.initialize(data_path)
    ModelSingleton.get(model_path, device, VocabSingleton.vocab_size())
