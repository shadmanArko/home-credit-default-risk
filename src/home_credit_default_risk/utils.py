"""Shared helpers used across feature modules."""

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """`numerator / denominator`, with zero/missing denominators and any
    resulting +/-inf mapped to `NaN`. Downstream imputers handle `NaN`
    uniformly; none of them handle infinities."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)
