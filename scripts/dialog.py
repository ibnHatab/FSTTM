
import datetime
import itertools
import time

from fsttm import Dialog

def select_min_cost(costs):
    return min(costs, key=costs.get)

class SystemProxy():
    def system(self):
        print(f'\t\t ++ SYSTEM_CB: SPEAK')

    def system_cb(self, system, flooor):
        print(f'\t\t >> SYSTEM_CB: {system}, {flooor}')

class UserProxy():
    def user(self):
        print(f'\t\t ++ USER_CB: SPEAK')

    def user_cb(self, user, flooor):
        print(f'\t\t >> USER_CB: {user}, {flooor}')

u = UserProxy()
s = SystemProxy()
m = Dialog(system_cb=s.system_cb, user_cb=u.user_cb)

dialog = [
    (lambda c: 1, 1),
    (lambda c: m.user_action('G'), 2),
    (lambda c: m.system_action(select_min_cost(c)), 0),
    (lambda c: m.user_action('R'), 1),
    (lambda c: m.system_action('G'), 3),
    (lambda c: m.system_action('R'), 3),
    (lambda c: 1, 1),
    #(lambda: m.system_action('G'), 3),
]

# Define a starting timestamp
start_timestamp = datetime.datetime.now()
duration = 0
# Infinite loop generating an index and timestamp
for index, timestamp in enumerate(itertools.count()):
    current_time = start_timestamp + datetime.timedelta(seconds=index)

    sa_cost = m.system_actions_cost()
    print(f'{index} - {m.state}:{m.system, m.user} \t(sa cost:{sa_cost})')

    if duration <= 0 and dialog:
        action, duration = dialog.pop(0)
        action(sa_cost)

    duration -= 1

    # Simulate some processing time
    time.sleep(.3)  # Adjust this delay as needed
    if not dialog:
        break
