##
# This script implements a non-deterministic finite automaton (NFA) for:
# A Finite-State Turn-Taking Model for Spoken Dialog Systems
##
# State transition function:
# SYSTEM, BOTHs, BOTHu, USER, FREEu, FREEs
#
# Cost of _system_ actions in each state:
# (K: keep the floor, R: release the floor, W : wait without the floor,
#  G: grab the floor, τ : time spent in current state, -: action unavailable)
##
# 1. The cost of an action that resolves either a gap or an overlap is zero
# 2. The cost of an action that creates unwanted gap or overlap is equal to a
#    constant parameter (potentially different for each action/state pair)
# 3. The cost of an action that maintains a gap or overlap is either a constant or
#    an increasing function of the total time spent in that state
##
#         | K      | R   | W      | G   | τ
# ------- |--------|-----|--------|-----| -
# SYSTEM  | 0      | C_s |   -    |  -  | -
# BOTHs   | C_o(t) | 0   |   -    |  -  | -
# BOTHu   | C_o(t) | 0   |   -    |  -  | -
# USER    |   -    |  -  | 0      | C_u | -
# FREEu   |   -    |  -  | C_g(t) | 0   | -
# FREEs   |   -    |  -  | C_g(t) | 0   | -
#
# C_s - is the cost of interrupting a system prompt before its end when the user
#       is not claiming the floor (false interruption)
# C_o(τ) - is the cost of remaining in an overlap that is already τ ms long
# C_u  - is the cost of grabbing the floor when the user is holding it (cut-in)
# C_g(τ) - is the cost of remaining in a gap that is already τ ms long
#
##

import math
import time
from collections import namedtuple, OrderedDict

Event = namedtuple('Event', ['event', 'src', 'dst'])

class Automaton():

    def __init__(self, initial_state=None, transitions=None):
        self.state = initial_state
        self.events = [Event(event, src, dst) for (event, src, dst) in transitions]
        self.prev_state = None
        self.state_start_time = int(round(time.time() * 1000))

    def trigger(self, e, predicate=None):
        """ Trigger an event with optional predicate
        """
        for tr in self.events:
            if tr.event == e and tr.src== self.state:
                ret = True
                if predicate is not None:
                    ret = predicate(tr)
                if ret:
                    self.prev_state = self.state
                    self.state = tr.dst
                    self.state_start_time = int(round(time.time() * 1000))
                    self.onchangestate(tr)
                return ret
        raise Exception(f'Invalid event {e} for state {self.state}')

    #abstractmethod
    def onchangestate(self, e):
        pass

    def in_state(self, t0):
        ti = int(round(time.time() * 1000)) - t0
        return ti

class Model(Automaton):
    su = lambda s, u: f'{s}_{u}' # (System_User) action
    _transitions = [
            (su('R', 'W'), 'SYSTEM', 'FREEs'),
            (su('G', 'W'), 'FREEs',  'SYSTEM'),
            (su('K', 'G'), 'SYSTEM', 'BOTHs'),
            (su('K', 'R'), 'BOTHs',  'SYSTEM'),
            (su('K', 'R'), 'BOTHu',  'SYSTEM'),
            (su('G', 'W'), 'FREEu',  'SYSTEM'),

            (su('G', 'W'), 'SYSTEM',  'SYSTEM'),

            (su('W', 'R'), 'USER',   'FREEu'),
            (su('W', 'G'), 'FREEu',  'USER'),
            (su('G', 'K'), 'USER',   'BOTHu'),
            (su('R', 'K'), 'BOTHu',  'USER'),
            (su('R', 'K'), 'BOTHs',  'USER'),
            (su('W', 'G'), 'FREEs',  'USER'),

            (su('W', 'G'), 'USER',  'USER'),

    ]
    _initial_state = 'USER'


    def __init__(self, system_cb=None, user_cb=None):
        self.state = Model._initial_state
        super(Model, self).__init__(Model._initial_state, Model._transitions)
        self.system = 'W'
        self.user = 'W'
        self.system_cb = system_cb
        self.user_cb = user_cb

    def onchangestate(self, e):
        print(f'\t>> {e.src}:\t({e.event}) -> {e.dst}\t--- ({self.system_actions_cost()})')

    @property
    def is_system(self):
        return self.state in ['SYSTEM', 'BOTHs', 'BOTHu']

    @property
    def is_user(self):
        return self.state in ['USER', 'FREEs', 'FREEu']

    def system_action(self, action):
        flooor = self.is_system
        ret = self.trigger(Model.su(action, self.user))
        # remap system state
        if ret:
            self.system = {
                'R': 'W',
                'G': 'K',
                'K': 'K',
            }[action]

        # signal floor turn to the system
        if flooor != self.is_system and self.system_cb:
            self.system_cb(self.system, flooor)


    def user_action(self, action):
        flooor = self.is_user
        ret = self.trigger(Model.su(self.system, action))
        # remap user state
        if ret:
            self.user = {
                'R': 'W',
                'G': 'K',
                'K': 'K',
            }[action]

        # signal floor turn to the user
        if flooor != self.is_user and self.user_cb:
            self.user_cb(self.system, flooor)

    def system_actions_cost(self):
        """
        Participants in a conversation attempt to
        minimize gaps and overlaps
        """
        # model parameters
        C_s = 100
        C_o = lambda tau: math.exp((tau+100)/1000)
        P_B = 0.1 # probability of being in a both state (regressor)

        C_u = 5000       # cost to grab in user
        C_g_pause = 1    # cost to grab in pause
        P_F_pause = 0.38 # probability of being in a free state (regressor)

        C_g_speech = 500  # cost to grab in speech
        P_F_speech = 0.20 # probability of being in a free state (regressor)

        tau = self.in_state(self.state_start_time)
        if self.is_system:
            return {
                'K': P_B * C_o(tau), # cost of overlap
                'R': (1-P_B)*C_s,    # cost of reducing gap
            }
        else: # is_user
            if self.state == 'FREEu': # At pause
                # cost proportional to the duration of the pause so far
                C_g = lambda tau: C_g_pause * tau
                P_F = P_F_pause
            elif self.state == 'USER': # In speech
                C_g = lambda tau: C_g_speech
                P_F = P_F_speech
            else:
                P_F = 0
                C_g = lambda tau: 1000 # in free system state
            return {
                'W': P_F*C_g(tau),
                'G': (1-P_F)*C_u,
            }


if __name__ == "__main__":
    ...