"""imagegen - AIVA's raster painting (the MOSS-Image sidecar, extracted).

Owns the 'image' canvas action kind: it registers itself into the canvas
language and provides the paint instructions.  Canvas stays vector-only and
imagegen-agnostic; agents that want raster painting import this package.
"""
from .moss import MossClient, MossPaintError, paint_size

try:
    from .bridge import PAINT_INSTRUCTIONS, register_image_kind
except ImportError:  # canvas not installed - painting without the language
    PAINT_INSTRUCTIONS = ""
    def register_image_kind():
        return None

__all__ = ['MossClient', 'MossPaintError', 'paint_size',
           'PAINT_INSTRUCTIONS', 'register_image_kind']
__version__ = '0.1.0'
