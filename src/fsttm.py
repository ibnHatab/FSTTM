
from fysom import FysomGlobalMixin, FysomGlobal


class Model(FysomGlobalMixin, object):
    """
    Finite-State Turn-Taking Machine (FSTTM), uses the same six state

    - USER and SYSTEM represent states where one and only one of the participants claims the floor,
    - FREEs and FREEu states where no participant claims the floor 
    - BOTHs and BOTHu states where both participants claim the floor 

    """
    GSM = FysomGlobal(
        events=[('warn',  'green',  'yellow'),
                {
                    'name': 'panic',
                    'src': ['green', 'yellow'],
                    'dst': 'red',
                    'cond': [  # can be function object or method name
                        'is_angry',  # by default target is "True"
                        {True: 'is_very_angry', 'else': 'yellow'}
                    ]
        },
            ('calm',  'red',    'yellow'),
            ('clear', 'yellow', 'green')],
        initial='green',
        final='red',
        state_field='state'
    )

    def __init__(self):
        self.state = None
        super(Model, self).__init__()

    def is_angry(self, event):
        return True

    def is_very_angry(self, event):
        return False


obj = Model()
obj.current  # 'green'
obj.warn()
obj.is_state('yellow')  # True
# conditions and conditional transition
obj.panic()
obj.current  # 'yellow'
obj.is_finished()  # False
