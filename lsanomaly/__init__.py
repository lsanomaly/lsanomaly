import logging

from lsanomaly import _lsanomaly
from lsanomaly._lsanomaly import LSAnomaly  # noqa
from lsanomaly import lengthscale_approx as lengthscale_approx

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [_lsanomaly, lengthscale_approx]
