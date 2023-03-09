import numpy as np
from PIL import Image

_DEFAULT_ORDER = 'HWC'


def _get_hwc_map(order_: str):
    order_ = order_.upper()
    return tuple(order_.index(c) for c in _DEFAULT_ORDER.upper())


_float_types = [np.float16, np.float32, np.float64]
if hasattr(np, 'float128'):
    _float_types.append(np.float128)
_float_types = tuple(_float_types)


def rgb_decode(data, order_: str = 'CHW') -> Image.Image:
    if data.dtype in (np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.uint64, np.int64):
        data = data.astype(np.uint8)
    elif data.dtype in _float_types:
        data = np.clip(data, 0.0, 1.0)
        data = (data * 255).astype(np.uint8)
    else:
        raise TypeError(f'Unknown dtype for data - {data.dtype!r}.')  # pragma: no cover

    data = np.transpose(data, _get_hwc_map(order_))
    return Image.fromarray(data, 'RGB')
