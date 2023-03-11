import glob
import os.path
import random

import numpy as np
import pytest
from PIL import Image
from hbutils.random import keep_global_state, global_seed
from hbutils.testing import tmatrix, disable_output
from sklearn.metrics import adjusted_rand_score

from imgutils.clustering.lpips import _image_encode, lpips_difference, lpips_clustering
from zoo.lpips.dispatch import _TRANSFORM
from ..testings import get_testfile


@keep_global_state()
def sampling_from_dataset(seed):
    global_seed(seed)
    files = sorted([file for file in glob.glob(os.path.join(get_testfile('dataset'), '**', '*'), recursive=True)
                    if os.path.isfile(file) and file.endswith('.jpg')])

    sample = random.sample(files, 50)
    t_exists = {}
    tn = 0
    answer = []
    for p in [os.path.basename(os.path.dirname(file)) for file in sample]:
        if p == 'free':
            tn += 1
            answer.append(tn)
        else:
            if p not in t_exists:
                tn += 1
                t_exists[p] = tn
                answer.append(tn)
            else:
                answer.append(t_exists[p])

    return sample, answer


@pytest.mark.unittest
class TestClusteringLpips:
    @pytest.mark.parametrize(*tmatrix({
        'filename': ['6124220.jpg', '6125785.png', '6125901.jpg'],
    }))
    def test_image_encode(self, filename):
        image = Image.open(get_testfile(filename))
        assert np.isclose(_image_encode(image), _TRANSFORM(image).numpy()).all()

    @pytest.mark.parametrize(*tmatrix({
        'f1': ['6124220.jpg', '6125785.png', '6125901.jpg'],
        'f2': ['6124220.jpg', '6125785.png', '6125901.jpg'],
    }))
    def test_lpips_difference(self, f1, f2):
        i1 = Image.open(get_testfile(f1))
        i2 = Image.open(get_testfile(f2))

        if f1 == f2:
            assert lpips_difference(i1, i2) == pytest.approx(0.0)
        else:
            assert lpips_difference(i1, i2) >= 0.55

    @pytest.mark.parametrize(*tmatrix({
        'seed': [0, 1, 2, 3, 4],
    }))
    def test_fuck(self, seed):
        files, answer = sampling_from_dataset(seed)
        with disable_output():
            data = lpips_clustering(files, threshold=0.45)

        p_exists = {}
        preds = []
        pn = 0
        for v in data:
            if v == -1:
                pn += 1
                preds.append(pn)
            else:
                if v not in p_exists:
                    pn += 1
                    p_exists[v] = pn
                    preds.append(pn)
                else:
                    preds.append(p_exists[v])

        assert adjusted_rand_score(answer, preds) >= 0.90