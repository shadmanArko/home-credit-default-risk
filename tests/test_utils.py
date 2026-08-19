import numpy as np
import pandas as pd

from home_credit_default_risk.utils import safe_divide


def test_normal_division():
    result = safe_divide(pd.Series([10.0, 20.0]), pd.Series([2.0, 4.0]))
    assert list(result) == [5.0, 5.0]


def test_zero_denominator_is_nan():
    result = safe_divide(pd.Series([10.0]), pd.Series([0.0]))
    assert pd.isna(result.iloc[0])


def test_missing_operand_propagates():
    result = safe_divide(pd.Series([np.nan]), pd.Series([2.0]))
    assert pd.isna(result.iloc[0])


def test_no_infinities_in_output():
    result = safe_divide(pd.Series([1.0, -1.0, 0.0]), pd.Series([0.0, 0.0, 0.0]))
    assert not np.isinf(result.to_numpy()).any()
