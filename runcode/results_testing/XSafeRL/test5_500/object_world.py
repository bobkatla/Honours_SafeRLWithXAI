import params as p
import pandas as pd
import numpy as np
from hazard import Hazard

world_object = np.empty([p.a, p.b], dtype=Hazard)
fire_1 = np.array([3,2])
fire_2 = np.array([3,8])
water_1 = np.array([7,2])
water_2 = np.array([1,6])

for x in range(len(p.world[0])):
    for y in range(len(p.world[1])):
        if p.world[x, y] == 1:
            world_object[x, y] = Hazard(1, 0, 0, 1)
        elif p.world[x, y] == 2:
            world_object[x, y] = Hazard(2, 80, 0, 0)
        elif p.world[x, y] == 3:
            world_object[x, y] = Hazard(3, 20, 100, 0)
        else:
            temp_dis = min(np.linalg.norm(np.array([x, y]) - fire_1), np.linalg.norm(np.array([x, y]) - fire_2))
            humid_dis = min(np.linalg.norm(np.array([x, y]) - water_1), np.linalg.norm(np.array([x, y]) - water_2))
            world_object[x, y] = Hazard(0, 80/temp_dis, 100/humid_dis, 0)



# start to create my own world
# What I wanna to do here is create the world with real time data and then use the agent to collect data
# The agent will then output the data for training. maybe human will help label as well
# Need to think about the input
# Input: tempt, humid, wall, x pos, y pos, action
# Will need to clarify that it is taking numeric data now, so will need to normalize/standardize the data
