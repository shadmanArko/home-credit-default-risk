from home_credit_default_risk.domain.scoring import DEFAULT_THRESHOLD, decide


def test_probability_at_threshold_is_high_risk():
    decision = decide(DEFAULT_THRESHOLD)
    assert decision.is_high_risk is True


def test_probability_just_below_threshold_is_not_high_risk():
    decision = decide(DEFAULT_THRESHOLD - 0.001)
    assert decision.is_high_risk is False


def test_probability_and_threshold_are_recorded_on_the_decision():
    decision = decide(0.9, threshold=0.5)
    assert decision.probability == 0.9
    assert decision.threshold == 0.5
    assert decision.is_high_risk is True


def test_custom_threshold_overrides_the_default():
    decision = decide(0.3, threshold=0.2)
    assert decision.is_high_risk is True
