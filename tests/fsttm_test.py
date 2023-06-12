import os
import sys
import pytest

from gpt_fsttm_server.fsttm import Model

#https://aclanthology.org/N09-1071.pdf
# 3.1 Extending the 6-state model for control

# Turn transitions with gap
def test_user_to_system():
    # USER (R,W ) → FREEU(G,W) → SYSTEM
    m = Model()
    assert m.state == 'USER'

    m.user_action('R')
    m.system_action('G')

    assert m.state == 'SYSTEM'

def test_system_to_user():
    # SYSTEM (R,W ) → FREES(W,G → USER
    m = Model()
    m.state = 'SYSTEM'
    assert m.state == 'SYSTEM'

    m.system_action('R')
    m.user_action('G')

    assert m.state == 'USER'


def test_user_to_free_user_and_back():
    # USER -> (R,W ) → FREEU -> (W,G) →  USER
    m = Model()
    assert m.state == 'USER'
    m.user_action('R')
    assert m.state == 'FREEu'
    m.user_action('G')
    assert m.state == 'USER'


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)] + sys.argv[1:])