'''
Using the greedy algorithm to solve the 
activity selection problem.
'''

import random

class Activity:
    '''
    This class represents the activty object.
    each activity has a name, starttime and endtime.
    '''
    def __init__(self, starttime: int, endtime: int, name: str):
        self.starttime  = starttime
        self.endtime    = endtime
        self.name       = name    

    def __repr__(self):
        return f"{self.name}: starttime: {self.get_starttime()} - endtime: {self.get_endtime()}"

    def get_starttime(self) -> int:
        return self.starttime
    
    def get_endtime(self) -> int:
        return self.endtime
    
    def get_name(self) -> str:
        return self.name


def isCompartible(activity1: Activity, other_activity: Activity) -> bool:
    '''
    Checks if two activities are compartible.
    condition: finishing time of activity 1 is less or equal to start time of other activity
    '''
    return activity1.get_endtime() <= other_activity.get_starttime()



# NOTE UNCOMMENT THE FOLLOWING BLOCK IF YOU WANT TO GENERATE THE ACTIVIES RANDOMLY
# activities = []
# # generate 15 activities
# for i in range(15):
#     minrange = 0
#     maxrange = 15
#     start = random.randint(minrange, maxrange)
#     end = random.randint(start+1, maxrange*2) # to ensure start is less than end time
#     activities.append(Activity(start, end, f"A{i+1}"))


activities = [
    Activity(1, 4, "A1"),
    Activity(3, 5, "A2"),
    Activity(0, 6, "A3"),
    Activity(5, 7, "A4"),
    Activity(3, 9, "A5"),
    Activity(5, 9, "A6"),
    Activity(6, 10, "A7"),
    Activity(8, 11, "A8"),
    Activity(8, 12, "A9"),
    Activity(2, 14, "A10"),
    Activity(12, 16, "A11"),
]

# sort activities to mimick greediness.
activities.sort(key=lambda activity: activity.get_endtime())
a1 = activities[0]
subset = [a1] # stores the optimal solution

for i in range(1, len(activities)):
    if isCompartible(subset[-1], activities[i]):
        subset.append(activities[i])

print("ACTIVITIES LIST")
for i in activities:
    print(f"({i})")

print("LARGEST SUBSET")
for i in subset:
    print(f"({i})")