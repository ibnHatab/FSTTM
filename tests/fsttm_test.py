import os
import sys
import pytest
import inspect

from fsttm import Model

# run all tests
def trace(): print("\n"+inspect.stack()[1][3])
if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)] + sys.argv[1:] + ['-s'])


#https://aclanthology.org/N09-1071.pdf
# 3.1 Extending the 6-state model for control

# _Turn transitions with gap_ are the most common
# way the floor goes from one participant to the
# other. For example, at the end of a user utter-
# ance, once the user finishes speaking, the floor
# becomes free, after which the system starts re-
# sponding, thus grabbing the floor.

def test_system_to_user():
    trace()
    # SYSTEM: (R,W) → FREEs:(W,G) → USER
    m = Model()
    m.state = 'SYSTEM'
    m.system_action('R')
    m.user_action('G')
    assert m.state == 'USER'

def test_user_to_system():
    trace()
    # USER: (R,W ) → FREEu: (G,W) → SYSTEM
    m = Model()
    m.state == 'USER'
    m.user_action('R')
    m.system_action('G')
    assert m.state == 'SYSTEM'

# _Turn transitions with overlap_ happen when a par-
# ticipant grabs the floor while it still belongs to
# the other. For example, when a user barges in
# on a system prompt, both participants hold the
# floor. Then, the system recognizes the barge-
# in attempt and relinquishes the floor, which be-
# comes user’s

def test_user_barge_in_on_system():
    trace()
    # SYSTEM: (K,G) → BOTHs: (R,K → USER
    m = Model()
    m.state = 'SYSTEM'
    m.system_action('G')
    m.user_action('G')
    m.system_action('R')
    assert m.state == 'USER'


def test_system_interrupts_user():
    trace()
    # USER: (G,K) → BOTHu: (K,R) → SYSTEM
    m = Model()
    m.user_action('G')
    m.system_action('G')
    m.user_action('R')
    assert m.state == 'SYSTEM'

# _Failed interruptions_ happen when a participant
# barges in on the other and then withdraws be-
# fore the original floor holder releases the floor.

def test_system_interrupts_user_but_detects_it():
    trace()
    # USER: (G,K) → BOTHu: (R,K) → USER
    m = Model()
    m.user_action('G')
    m.system_action('G')
    m.system_action('R')
    print('1>> ', m.system, m.user)
    assert m.state == 'USER'

# user backchannels a system utterance
def test_system_failing_to_react_fast_enough():
    trace()
    # SYSTEM: (K,G) → BOTHs: (K,R) → SYSTEM
    m = Model()
    m.state = 'SYSTEM'
    m.system_action('G')
    m.user_action('G')
    m.user_action('R')
    print('2>> ', m.system, m.user)
    assert m.state == 'SYSTEM'


# _Time outs_ start like transitions with gap but the intended next speaker
# does not take the floor and the original floor holder grabs it back.
def test_system_attempts_to_reestablish_communication():
    trace()
    # SYSTEM: (R,W ) → FREEs: (G,W → SYSTEM
    m = Model()
    m.state = 'SYSTEM'
    m.system_action('G')
    m.system_action('R')
    # timeout
    m.system_action('G')
    print('3>> ', m.system, m.user)
    assert m.state == 'SYSTEM'


def test_system_to_slow_to_respond_to_user():
    trace()
    # USER: (W,R) → FREEu: (W,G → USER
    m = Model()
    m.user_action('G')
    m.user_action('R')
    # timeout
    m.user_action('G')
    print('4>> ', m.system, m.user)
    assert m.state == 'USER'


# Self transitions happen when a participant keeps the floor for a long time
def test_user_to_free_user_and_back():
    trace()
    # USER -> (R,W ) → FREEu -> (W,G) →  USER
    m = Model()
    assert m.state == 'USER'
    m.user_action('R')
    assert m.state == 'FREEu'
    m.user_action('G')
    assert m.state == 'USER'
