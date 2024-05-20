import os
from tqdm.auto import tqdm as original_tqdm
import functools
import logging as log


tqdm_disabled = os.environ.get('DISABLE_TQDM', False)
if tqdm_disabled:
    log.info("Disabling tqdm")
tqdm = original_tqdm
tqdm.__init__ = functools.partialmethod(
    tqdm.__init__, mininterval=0.5, disable=tqdm_disabled)
