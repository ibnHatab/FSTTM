
from collections import namedtuple

Event = namedtuple('Event', ['event', 'src', 'dst'])

class Automaton():

    def __init__(self, initial_state=None, transitions=None):
        self.state = initial_state
        self.events = [Event(event, src, dst) for (event, src, dst) in transitions]

    def trigger(self, e):
        """ Trigger an event.
        """
        for tr in self.events:
            if tr.name == e and tr.src== self.state:
                self.state = tr.dst
                self.onchangestate(e)
                return
        raise Exception(f'Invalid event {e} for state {self.state}')

    #abstractmethod
    def onchangestate(self, e):
        pass

class Model(Automaton):
    su = lambda s, u: f'{s}_{u}' # system, user action
    _transitions = [
#           ('event', 'src', 'dst'),
            (su('R', 'W'), 'SYSTEM', 'FREEs'),
            (su('G', 'W'), 'FREEs',  'SYSTEM'),
            (su('K', 'G'), 'SYSTEM', 'BOTHs'),
            (su('K', 'R'), 'BOTHs',  'SYSTEM'),
            (su('K', 'R'), 'BOTHu',  'SYSTEM'),
            (su('G', 'W'), 'FREEu',  'SYSTEM'),

            (su('W', 'R'), 'USER',   'FREEu'),
            (su('W', 'G'), 'FREEu',  'USER'),
            (su('G', 'K'), 'USER',   'BOTHu'),
            (su('R', 'K'), 'BOTHu',  'USER'),
            (su('R', 'K'), 'BOTHs',  'USER'),
            (su('W', 'G'), 'FREEs',  'USER'),
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
        print(f'>> {e.event} {e.src} -> {e.dst}: ({self.system}, {self.user})')

    @property
    def is_system(self):
        return self.state in ['SYSTEM', 'BOTHs', 'BOTHu']

    @property
    def is_user(self):
        return self.state in ['USER', 'FREEs', 'FREEu']

    def system_action(self, action):
        flooor = self.is_system
        self.trigger(Model.su(action, self.user))

        # signal floor turn to the system
        if flooor != self.is_system and self.system_cb:
            self.system_cb(self.system, flooor)

        # remap system state
        self.system = {
            'R': 'W',
            'G': 'K',
        }[action]

    def user_action(self, action):
        flooor = self.is_user
        self.trigger(Model.su(self.system, action))

        # signal floor turn to the user
        if flooor != self.is_user and self.user_cb:
            self.user_cb(self.system, flooor)

        # remap system state
        self.system = {
            'R': 'W',
            'G': 'K',
        }[action]
