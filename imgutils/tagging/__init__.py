"""
Overview:
    Get tags for anime images.

    This is an overall benchmark of all the danbooru models:

    .. image:: benchmark_tagging.bm.svg
        :align: center

"""
from .deepdanbooru import get_deepdanbooru_tags
from .format import tags_to_text
from .wd14 import get_wd14_tags
