"""CPU smoke test of hand_defects.hadm_scorer with a fake HADM (and the real hand detector if HAND_FIXER_TEST_YOLO is set)."""
import numpy as np

from hand_defects import hadm_scorer as hs


class _FakeHands:
    def __call__(self, rgb, conf=0.3):
        return [([10.0, 10.0, 30.0, 30.0], 0.9), ([60.0, 60.0, 80.0, 80.0], 0.8)]


class _FakeHADM:
    def __call__(self, rgb):
        return [dict(box=[12, 12, 28, 28], score=0.9, cls="hand"), dict(box=[60, 60, 80, 80], score=0.95, cls="face")]


def test_scorer_rule_and_calibration():
    s = hs.HADMHandScorer.__new__(hs.HADMHandScorer)
    s.hands, s.hadm, s.a, s.b, s.min_inside, s.pad = _FakeHands(), _FakeHADM(), 4.108, -4.039, 0.5, 1.3
    out = s(np.zeros((100, 100, 3), np.uint8), threshold=0.2)
    assert [h.hadm_raw for h in out] == [0.9, 0.0]          # face artifact is ignored; box must be >=50% inside
    assert out[0].bad and not out[1].bad
    assert abs(out[0].p_bad - 1 / (1 + np.exp(-(4.108 * 0.9 - 4.039)))) < 1e-3
