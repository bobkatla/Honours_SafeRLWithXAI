import os
import numpy as np
import params as p
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches
from object_world import world_object

df = pd.DataFrame(None, columns=['temp', 'humid', 'wall', 'x', 'y', 'action', 'label'])

def Qlearn_multirun_tab():
    # This function just runs multiple instances of
    # Q-learning. Doing so helps obtain an average performance
    # measure over multiple runs.
    retlog = []  # log of returns of all episodes, in all runs
    for i in range(p.Nruns):
        print("Run no:", i)
        Q, ret = main_Qlearning_tab()  # call Q learning
        if i == 0:
            retlog = ret
        else:
            retlog = np.vstack((retlog, ret))
        # retlog.append(ret)
        if (i + 1) / p.Nruns == 0.25:
            print('25% runs complete')
        elif (i + 1) / p.Nruns == 0.5:
            print('50% runs complete')
        elif (i + 1) / p.Nruns == 0.75:
            print('75% runs complete')
        elif (i + 1) == p.Nruns:
            print('100% runs complete')
    # meanreturns=(np.mean(retlog,axis=0))
    return Q, retlog


def main_Qlearning_tab():
    # This calls the main Q learning algorithm
    # This Q is the 3D matrix for the Q storing (States x Actions)
    Q = np.zeros((p.a, p.b, p.A))  # initialize Q function as zeros
    returns = []  # stores returns for each episode
    for i in range(p.episodes):
        if (i + 1) / p.episodes == 0.25:
            print('25% episodes done')
        elif (i + 1) / p.episodes == 0.5:
            print('50% episodes done')
        elif (i + 1) / p.episodes == 0.75:
            print('75% episodes done')
        elif (i + 1) / p.episodes == 1:
            print('100% episodes done')
        Q, ret = Qtabular(Q, i)  # call Q learning
        if i % 1 == 0:
            returns.append(
                ret)  # compute return offline- can also be done online, but this way, a better estimate can be obtained
    return Q, returns


def Qtabular(Q, episode_no):
    initial_state = np.array([(p.a - 1) * np.random.random_sample(), (p.b - 1) * np.random.random_sample()])
    rounded_initial_state = staterounding(initial_state)
    while p.world[rounded_initial_state[0], rounded_initial_state[1]] == 1:
        initial_state = np.array([(p.a - 1) * np.random.random_sample(), (p.b - 1) * np.random.random_sample()])
        rounded_initial_state = staterounding(initial_state)
    state = staterounding(initial_state.copy())
    count = 0
    breakflag = 0
    eps_live = 1 - (p.epsilon_decay * episode_no)
    ret = 0
    target_state = p.targ
    for i in range(p.breakthresh):

        count = count + 1
        if breakflag == 1:
            break
        if count > p.breakthresh:
            breakflag = 1
        if eps_live > np.random.sample():
            a = np.random.randint(p.A)
        else:
            Qmax, Qmin, a = maxQ_tab(Q, state)

        next_state = transition(state, a)

        roundedstate = staterounding(state)

        # print("start the df append")
        cor_object = world_object[state[0], state[1]]
        next_cor_object = world_object[next_state[0], next_state[1]]
        new_data = [cor_object.temp, cor_object.humid, cor_object.wall, state[0], state[1], a, next_cor_object.label]
        df.loc[len(df)] = new_data
        if len(df) == 21000:
            df.to_csv('full.csv', index=False)
            print("break NOW")

        if p.world[next_state[0], next_state[1]] == 0 and (
                p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
            if np.linalg.norm(next_state - target_state) <= p.thresh:
                R = p.highreward
            else:
                R = p.livingpenalty
        elif p.world[next_state[0], next_state[1]] == 2 and (
                p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
            # When there is fire
            R = p.firePelnaty
            next_state = state.copy()
        elif p.world[next_state[0], next_state[1]] == 3 and (
                p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
            # When there is water
            R = p.waterPelnaty
            next_state = state.copy()
        else:
            R = p.penalty
            next_state = state.copy()

        ret = ret + R

        Qmaxnext, Qminnext, aoptnext = maxQ_tab(Q, next_state)
        Qtarget = R + (p.gamma * Qmaxnext) - Q[roundedstate[0], roundedstate[1], a]
        Q[roundedstate[0], roundedstate[1], a] = Q[roundedstate[0], roundedstate[1], a] + (p.alpha * Qtarget)
        if np.linalg.norm(next_state - target_state) <= p.thresh:
            break
        state = next_state.copy()

    return Q, ret


def maxQ_tab(Q, state):
    # get max of Q values and corresponding action
    Qlist = []
    roundedstate = staterounding(state)
    for i in range(p.A):
        Qlist.append(Q[roundedstate[0], roundedstate[1], i])
    tab_maxQ = np.max(Qlist)
    tab_minQ = np.min(Qlist)
    maxind = []
    for j in range(len(Qlist)):
        if tab_maxQ == Qlist[j]:
            maxind.append(j)
    # print(maxind)
    if len(maxind) > 1:
        optact = maxind[np.random.randint(len(maxind))]
    else:
        optact = maxind[0]
    return tab_maxQ, tab_minQ, optact


def transition(state, act):
    # print(orig_state)
    # print(act)
    n1 = np.random.uniform(low=-0.2, high=0.2, size=(1,))  # x noise
    n2 = np.random.uniform(low=-0.2, high=0.2, size=(1,))  # y noise
    new_state = state.copy()
    if act == 0:
        new_state[0] = state[0]
        new_state[1] = state[1] + 1  # move up
    elif act == 1:
        new_state[0] = state[0] + 1  # move right
        new_state[1] = state[1]
    elif act == 2:
        new_state[0] = state[0]
        new_state[1] = state[1] - 1  # move down
    elif act == 3:
        new_state[0] = state[0] - 1  # move left
        new_state[1] = state[1]

    # new_state[0]=new_state[0]+n1
    # new_state[1]=new_state[1]+n2
    return new_state


def staterounding(state):
    # rounds off states
    roundedstate = [0, 0]
    roundedstate[0] = int(np.around(state[0]))
    roundedstate[1] = int(np.around(state[1]))
    if roundedstate[0] >= (p.a - 1):
        roundedstate[0] = p.a - 2
    elif roundedstate[0] < 1:
        roundedstate[0] = 1
    if roundedstate[1] >= (p.b - 1):
        roundedstate[1] = p.b - 2
    elif roundedstate[1] <= 0:
        roundedstate[1] = 1
    return roundedstate


#######################################
if __name__ == "__main__":
    try:
        Q, retlog = Qlearn_multirun_tab()
        # df.to_csv('training_data.csv', index=False)
    except KeyboardInterrupt:
        print("break")
        # df.to_csv('training_data.csv', index=False)
