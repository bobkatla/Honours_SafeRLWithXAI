import numpy as np
import matplotlib.pyplot as plt

EP = 1000
n_test = 5
type_test = ['100', '500', '1000', '2000']
# this is pair color match with each group
color_gr = ['darkorange', 'lightcoral', 'mediumpurple', 'lightskyblue']
n_gr = len(type_test)

# each of them below would be 4 x 5 x 10 x 1000 (4 dim)
ha_count_Q = [[] for _ in type_test]
ha_count_SARSA = [[] for _ in type_test]
re_Q = [[] for _ in type_test]
re_SARSA = [[] for _ in type_test]

for i in range(n_gr):
    for j in range(n_test):
        ha_count_Q[i].append(np.load(f'./test{j + 1}_{type_test[i]}/Q_ha.npy'))
        ha_count_SARSA[i].append(np.load(f'./test{j + 1}_{type_test[i]}/SARSA_ha.npy'))
        re_Q[i].append(np.load(f'./test{j + 1}_{type_test[i]}/Q_re.npy'))
        re_SARSA[i].append(np.load(f'./test{j + 1}_{type_test[i]}/SARSA_re.npy'))

def get_avg_each_ep(data, ep):
    hold = [run[ep] for run in data]
    return np.mean(hold)

def get_results(data_gr):
    arr_min = []
    arr_max = []
    arr_mean = []
    for i in range(EP):
        arr = [] # this will be 5 vals
        for a in data_gr:
            avg = get_avg_each_ep(a, i)
            arr.append(avg)
        arr_min.append(min(arr))
        arr_max.append(max(arr))
        arr_mean.append(np.mean(arr))
    return arr_min, arr_max, arr_mean

def set_plot_inside(a, X, min, max, mean, color, label):
    a.plot(X, mean, color=color, label=label)
    # a.fill_between(x=X, y1=min, y2=max, color=color, alpha=0.2)

def set_graph(X, data, ylable, pathFull, pathOri):
    f, a = plt.subplots()

    a.set_xlabel('Episodes', fontsize=20)
    a.set_ylabel(ylable, fontsize=25)

    data_full = np.load(pathFull)
    data_ori = np.load(pathOri)
    arr_full = []
    arr_ori = []
    for i in range(EP):
        avg_full = get_avg_each_ep(data_full, i)
        arr_full.append(avg_full)
        avg_ori = get_avg_each_ep(data_ori, i)
        arr_ori.append(avg_ori)

    for i in range(n_gr):
        min, max, mean = get_results(data[i])
        set_plot_inside(a, X, min, max, mean, color_gr[i], type_test[i])

    a.plot(X, arr_full, color='red', label='21,000')
    a.plot(X, arr_ori, color='black', label='original')

    a.legend()

#######################################
if __name__ == "__main__":
    X = range(1000)
    set_graph(X, ha_count_Q, 'Total hazard count - Q', './full/Q_ha.npy', './Q_ha.npy')
    set_graph(X, ha_count_SARSA, 'Total hazard count - SARSA', './full/SARSA_ha.npy', './SARSA_ha.npy')
    set_graph(X, re_Q, 'Reward - Q', './full/Q_re.npy', './Q_re.npy')
    set_graph(X, re_SARSA, 'Reward - SARSA', './full/SARSA_re.npy', './SARSA_re.npy')

    plt.show()
