"""Machine-checked verification of fsttm/fsttm.py against N09-1071 §3.1 —
the canonical transition table and the paper's structural constraints.
Mirrors go/internal/fsm/paper_test.go; see
docs/fsttm-n09-1071-verification.md."""
import pytest

from fsttm.fsttm import Model


def at(state):
    m = Model()
    m.onchangestate = lambda e: None
    # The model initializes with action vector (W,W) in USER; the paper's
    # USER state implies the user KEEPS the floor (vector (W,K)). A first
    # user grab (self-loop) normalizes it — exactly what the live engine's
    # first VAD onset does. See deviation E2 in the audit doc.
    steps = {
        'USER':   [(m.user_action, 'G')],
        'FREEu':  [(m.user_action, 'G'), (m.user_action, 'R')],
        'SYSTEM': [(m.user_action, 'R'), (m.system_action, 'G')],
        'FREEs':  [(m.user_action, 'R'), (m.system_action, 'G'),
                   (m.system_action, 'R')],
        'BOTHs':  [(m.user_action, 'R'), (m.system_action, 'G'),
                   (m.user_action, 'G')],
        'BOTHu':  [(m.user_action, 'G'), (m.system_action, 'G')],
    }[state]
    for fn, a in steps:
        fn(a)
    assert m.state == state
    return m


CANONICAL = [
    # (src, actor, action, dst) — the paper's 12 transitions
    ('SYSTEM', 'system', 'R', 'FREEs'),   # 1  release at prompt end
    ('FREEs',  'user',   'G', 'USER'),    # 2  gap transition
    ('FREEs',  'system', 'G', 'SYSTEM'),  # 3  time-out re-establish
    ('USER',   'user',   'R', 'FREEu'),   # 4  user releases
    ('FREEu',  'system', 'G', 'SYSTEM'),  # 5  gap transition
    ('FREEu',  'user',   'G', 'USER'),    # 6  user resumes
    ('SYSTEM', 'user',   'G', 'BOTHs'),   # 7  barge-in attempt
    ('BOTHs',  'system', 'R', 'USER'),    # 8  successful barge-in
    ('BOTHs',  'user',   'R', 'SYSTEM'),  # 9  failed user interruption
    ('USER',   'system', 'G', 'BOTHu'),   # 10 system cut-in
    ('BOTHu',  'user',   'R', 'SYSTEM'),  # 11 successful cut-in
    ('BOTHu',  'system', 'R', 'USER'),    # 12 failed system interruption
]


@pytest.mark.parametrize("src,actor,action,dst", CANONICAL)
def test_canonical_transition(src, actor, action, dst):
    m = at(src)
    (m.system_action if actor == 'system' else m.user_action)(action)
    assert m.state == dst


ILLEGAL = [
    # §3.1: no SYSTEM→BOTHu; releases/keeps from the wrong side
    ('SYSTEM', 'system', 'K'),
    ('USER',   'system', 'R'),
    ('FREEs',  'user',   'R'),
    ('FREEu',  'user',   'R'),
    ('FREEs',  'system', 'K'),
    ('BOTHs',  'user',   'G'),
    ('BOTHu',  'system', 'G'),
]


@pytest.mark.parametrize("src,actor,action", ILLEGAL)
def test_illegal_transition_raises(src, actor, action):
    m = at(src)
    with pytest.raises(Exception):
        (m.system_action if actor == 'system' else m.user_action)(action)


def test_cost_availability_matches_table1():
    for s in ('SYSTEM', 'BOTHs', 'BOTHu'):
        c = at(s).system_actions_cost()
        assert set(c) == {'K', 'R'}, s
    for s in ('USER', 'FREEu', 'FREEs'):
        c = at(s).system_actions_cost()
        assert set(c) == {'W', 'G'}, s
    # §4.1: in USER, grabbing (cut-in) must be far costlier than waiting
    c = at('USER').system_actions_cost()
    assert c['G'] > c['W']
