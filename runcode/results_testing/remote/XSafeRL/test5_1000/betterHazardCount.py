import os
import numpy as np
import params as p
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib.patches as patches
from hazard import Hazard
import pandas

from model_loading import hazard_prediction

from object_world import world_object


# import priors_tabular as PR
def Qlearn_multirun_tab():
    # This function just runs multiple instances of
    # Q-learning. Doing so helps obtain an average performance
    # measure over multiple runs.
    retlog = []  # log of returns of all episodes, in all runs
    hazlog = []
    for i in range(p.Nruns):
        print("Run no:", i)
        Q, ret, haz = main_Qlearning_tab()  # call Q learning
        if i == 0:
            retlog = ret
            hazlog = haz
        else:
            retlog = np.vstack((retlog, ret))
            hazlog = np.vstack((hazlog, haz))
        if (i + 1) / p.Nruns == 0.25:
            print('25% runs complete')
        elif (i + 1) / p.Nruns == 0.5:
            print('50% runs complete')
        elif (i + 1) / p.Nruns == 0.75:
            print('75% runs complete')
        elif (i + 1) == p.Nruns:
            print('100% runs complete')
    return Q, retlog, hazlog


def main_Qlearning_tab():
    # This calls the main Q learning algorithm
    # This Q is the 3D matrix for the Q storing (States x Actions)
    Q = np.zeros((p.a, p.b, p.A))  # initialize Q function as zeros
    goal_state = p.targ  # target point
    returns = []  # stores returns for each episode
    count_ha = []
    for i in range(p.episodes):
        if (i + 1) / p.episodes == 0.25:
            print('25% episodes done')
        elif (i + 1) / p.episodes == 0.5:
            print('50% episodes done')
        elif (i + 1) / p.episodes == 0.75:
            print('75% episodes done')
        elif (i + 1) / p.episodes == 1:
            print('100% episodes done')
        Q, ret, ha_no = Qtabular(Q, i)  # call Q learning
        if i % 1 == 0:
            # NOTE: right now just using sum of all the hazard
            count_ha.append(sum(ha_no))
            returns.append(ret)  # compute return offline- can also be done online, but this way, a better estimate can be obtained
    return Q, returns, count_ha


def Qtabular(Q, episode_no):
    initial_state = np.array([(p.a - 1) * np.random.random_sample(), (p.b - 1) * np.random.random_sample()])
    rounded_initial_state = staterounding(initial_state)
    while p.world[rounded_initial_state[0], rounded_initial_state[1]] == 1:
        initial_state = np.array([(p.a - 1) * np.random.random_sample(), (p.b - 1) * np.random.random_sample()])
        rounded_initial_state = staterounding(initial_state)
    state = staterounding(initial_state.copy())
    eps_live = 1 - (p.epsilon_decay * episode_no)
    ret = 0
    target_state = p.targ
    hazard_count = [0, 0, 0] # ha count with fire, water, wall
    for i in range(p.breakthresh):
        if eps_live > np.random.sample():
            a = np.random.randint(p.A)
        else:
            Qmax, Qmin, a = maxQ_tab(Q, state)

        # Check with DL
        cor_object = world_object[state[0], state[1]]
        check_hazard = hazard_prediction(cor_object.temp, cor_object.humid, cor_object.wall, state[0], state[1], a)

        roundedstate = staterounding(state)
        next_state = transition(state, a)
        is_hazard_next = True

        # hazard prediction
        if check_hazard == 1:
            # wall
            R = p.penalty
        elif check_hazard == 2:
            # fire
            R = p.firePelnaty
        elif check_hazard == 3:
            # water
            R = p.waterPelnaty
        else:
            # DL predicts safe, hence, make action
            if p.world[next_state[0], next_state[1]] == 0 and (
                    p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
                if np.linalg.norm(next_state - target_state) <= p.thresh:
                    R = p.highreward
                else:
                    R = p.livingpenalty
                is_hazard_next = False
            elif p.world[next_state[0], next_state[1]] == 2 and (
                    p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
                # When there is fire
                R = p.firePelnaty
                hazard_count[0] += 1
            elif p.world[next_state[0], next_state[1]] == 3 and (
                    p.a >= next_state[0] >= 0 and p.b >= next_state[1] >= 0):
                # When there is water
                R = p.waterPelnaty
                hazard_count[1] += 1
            else:
                R = p.penalty
                hazard_count[2] += 1

        # update Q-function
        Qmaxnext, Qminnext, aoptnext = maxQ_tab(Q, next_state)
        Qtarget = R + (p.gamma * Qmaxnext) - Q[roundedstate[0], roundedstate[1], a]
        Q[roundedstate[0], roundedstate[1], a] = Q[roundedstate[0], roundedstate[1], a] + (p.alpha * Qtarget)

        if is_hazard_next:
            next_state = state.copy()
        ret = ret + R

        if np.linalg.norm(next_state - target_state) <= p.thresh:
            print(f"I reached the goal at ep {episode_no}")
            break
        state = next_state.copy()

        if i == p.breakthresh - 1:
            print(f"I have not reached the goal at ep {episode_no}")

    return Q, ret, hazard_count


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


def optpol_visualize(Qp):
    for i in range(p.a):
        for j in range(p.b):
            if p.world[i, j] == 0:
                Qmaxopt, Qminopt, optact = maxQ_tab(Qp, [i, j])
                if optact == 0:
                    plt.scatter(i, j, color='red')
                elif optact == 1:
                    plt.scatter(i, j, color='green')
                elif optact == 2:
                    plt.scatter(i, j, color='blue')
                elif optact == 3:
                    plt.scatter(i, j, color='yellow')

    plotmap(p.world)
    plt.show()


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


########Additional functions for visualization######
def plotmap(worldmap):
    # plots the obstacle map
    for i in range(p.a):
        for j in range(p.b):
            if worldmap[i, j] > 0:
                plt.scatter(i, j, color='black')
    plt.show()


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


def opt_pol(Q, state, goal_state):
    # shows optimal policy
    plt.figure(0)
    plt.ion()
    for i in range(p.a):
        for j in range(p.b):
            if p.world[i, j] > 0:
                plt.scatter(i, j, color='black')
    plt.show()
    pol = []
    statelog = []
    count = 1
    while np.linalg.norm(state - goal_state) >= 1:
        Qm, Qmin, a = maxQ_tab(Q, state)
        if np.random.sample() > 0.9:
            a = np.random.randint(p.A)
        next_state = transition(state, a)
        roundednextstate = staterounding(next_state)
        if p.world[roundednextstate[0], roundednextstate[1]] == 1:
            next_state = state.copy()
        pol.append(a)
        statelog.append(state)
        print(state)
        plt.ylim(0, p.b)
        plt.xlim(0, p.a)
        plt.scatter(state[0], state[1], (60 - count * 0.4), color='blue')
        plt.draw()
        plt.pause(0.1)
        state = next_state.copy()
        print(count)
        if count >= 100:
            break
        count = count + 1
    return statelog, pol


def mapQ(Q):
    # plots a map of the value function
    fig = plt.figure(1)
    plt.ion
    Qmap = np.zeros((p.a, p.b))
    for i in range(p.a):
        for j in range(p.b):
            Qav = 0
            for k in range(p.A):
                Qav = Qav + Q[i, j, k]
            Qmap[i, j] = Qav
    # Qfig=plt.imshow(np.rot90(Qmap))
    Qmap = Qmap - np.min(Qmap)
    if np.max(Qmap) > 0:
        Qmap = Qmap / np.max(Qmap)
    Qmap = np.rot90(Qmap)

    return Qmap


#######################################
if __name__ == "__main__":
    # w,Qimall=Qlearn_main_vid()

    Q, reward_rlog, hazard_rlog = Qlearn_multirun_tab()

    np.save("Q_Q", Q)
    np.save("Q_ha", hazard_rlog)
    np.save("Q_re", reward_rlog)

    f1, a1 = plt.subplots()
    f2, a2 = plt.subplots()

    mr = (np.mean(hazard_rlog, axis=0))
    csr = []
    for i in range(len(mr)):
        if i > 0:
            csr.append(np.sum(mr[0:i]) / i)
    # np.savez("DQN" + str(p.Nruns) + "_runs_ha.npy.npz", hazard_rlog, Q)
    s_retlog = np.shape(hazard_rlog)
    x = range(s_retlog[1])
    mn = np.mean(hazard_rlog, axis=0)
    st_err = np.std(hazard_rlog, axis=0) / np.sqrt(p.Nruns)
    a1.set_xlabel('Episodes', fontsize=15)
    a1.set_ylabel('Hazard count total', fontsize=15)
    # a1.gca().legend(('Q-learning'), frameon=False)
    a1.grid(linestyle='-')
    a1.plot(x, mn, 'r')
    a1.fill_between(x, mn - st_err, mn + st_err, color='darksalmon', alpha=0.3)
    # plt.show()
    f1.savefig('better_ha_count_normal_RL.png')

    mr = (np.mean(reward_rlog, axis=0))
    csr = []
    for i in range(len(mr)):
        if i > 0:
            csr.append(np.sum(mr[0:i]) / i)
    # np.savez("DQN" + str(p.Nruns) + "_runs_re.npy.npz", reward_rlog, Q)
    s_retlog = np.shape(reward_rlog)
    x = range(s_retlog[1])
    mn = np.mean(reward_rlog, axis=0)
    st_err = np.std(reward_rlog, axis=0) / np.sqrt(p.Nruns)
    a2.set_xlabel('Episodes', fontsize=15)
    a2.set_ylabel('Reward', fontsize=15)
    # a2.gca().legend(('Q-learning'), frameon=False)
    a2.grid(linestyle='-')
    a2.plot(x, mn, 'r')
    a2.fill_between(x, mn - st_err, mn + st_err, color='darksalmon', alpha=0.3)
    # plt.show()
    f2.savefig('better_reward_normal_RL.png')
