from typing import Dict, Optional, Tuple

import numpy as np
from huggingface_hub import hf_hub_download

from imgutils.data import ImageTyping
from imgutils.generic import ClassifyModel

__all__ = [
    'anime_dbaesthetic',
]

_DEFAULT_MODEL_NAME = 'swinv2pv3_v0_448_ls0.2_x'
_REPO_ID = 'deepghs/anime_aesthetic'
_LABELS = ["worst", "low", "normal", "good", "great", "best", "masterpiece"]
_DEFAULT_LABEL_MAPPING = {
    'masterpiece': 0.95,
    'best': 0.85,
    'great': 0.75,
    'good': 0.5,
    'normal': 0.25,
    'low': 0.1,
    'worst': 0.0,
}


def _value_replace(v, mapping):
    if isinstance(v, (list, tuple)):
        return type(v)([_value_replace(vitem, mapping) for vitem in v])
    elif isinstance(v, dict):
        return type(v)({key: _value_replace(value, mapping) for key, value in v.items()})
    else:
        try:
            _ = hash(v)
        except TypeError:
            return v
        else:
            return mapping.get(v, v)


class AestheticModel:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.classifier = ClassifyModel(repo_id)
        self.cached_samples: Dict[str, Tuple] = {}

    def get_aesthetic_score(self, image: ImageTyping, model_name: str) -> Tuple[float, Dict[str, float]]:
        scores = self.classifier.predict_score(image, model_name)
        return sum(scores[label] * i for i, label in enumerate(_LABELS)), scores

    def _get_xy_samples(self, model_name: str):
        if model_name not in self.cached_samples:
            stacked = np.load(hf_hub_download(
                repo_id=self.repo_id,
                repo_type='model',
                filename=f'{model_name}/samples.npz',
            ))['arr_0']
            x, y = stacked[0], stacked[1]
            self.cached_samples[model_name] = ((x, x.min(), x.max()), (y, y.min(), y.max()))
        return self.cached_samples[model_name]

    def score_to_percentile(self, score: float, model_name: str):
        (x, x_min, x_max), (y, y_min, y_max) = self._get_xy_samples(model_name)
        idx = np.searchsorted(x, np.clip(score, a_min=x_min, a_max=x_max))
        if idx < x.shape[0] - 1:
            x0, y0 = x[idx], y[idx]
            x1, y1 = x[idx + 1], y[idx + 1]
            if np.isclose(x1, x0):
                return y[idx]
            else:
                return np.clip((score - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=y_min, a_max=y_max)
        else:
            return y[idx]

    @classmethod
    def percentile_to_label(cls, percentile: float, mapping: Optional[Dict[str, float]] = None):
        mapping = mapping or _DEFAULT_LABEL_MAPPING
        for label, threshold in sorted(mapping.items(), key=lambda x: (-x[1], x[0])):
            if percentile >= threshold:
                return label
        else:
            raise ValueError(f'No label for unknown percentile {percentile:.3f}.')

    def get_aesthetic(self, image: ImageTyping, model_name: str, fmt=('label', 'percentile')):
        score, confidence = self.get_aesthetic_score(image, model_name)
        percentile = self.score_to_percentile(score, model_name)
        label = self.percentile_to_label(percentile)
        return _value_replace(
            v=fmt,
            mapping={
                'label': label,
                'percentile': percentile,
                'score': score,
                'confidence': confidence,
            }
        )

    def clear(self):
        self.classifier.clear()
        self.cached_samples.clear()


_MODEL = AestheticModel(_REPO_ID)


def anime_dbaesthetic(image: ImageTyping, model_name: str = _DEFAULT_MODEL_NAME,
                      fmt=('label', 'percentile')):
    return _MODEL.get_aesthetic(image, model_name, fmt)
