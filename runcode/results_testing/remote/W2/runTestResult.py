import numpy as np
import matplotlib.pyplot as plt

EP = 1000
n_test = 5
type_test = ['100', '500', '1000', '2000']
# this is pair color match with each group
color_gr = ['darkorange', 'lightcoral', 'mediumpurple', 'lightskyblue']
n_gr = len(type_test)
# this is to smooth the light
smooth_factor = 10 # make sure diviable by EP
actual_n = int(EP/smooth_factor)

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
    # the data_gr is the whole group of test which is 5 x 10 x 1000
    arr_min = []
    arr_max = []
    arr_mean = []

    for i in range(actual_n):
        arr_outside = [] # this would be 10 vals
        for j in range(smooth_factor):
            actual_index = i*smooth_factor + j
            arr_inside = [] # this will be 5 vals
            for a in data_gr:
                avg = get_avg_each_ep(a, actual_index)
                arr_inside.append(avg)
            arr_outside.append(np.mean(arr_inside))

        val_avg = np.mean(arr_outside)
        val_std = np.std(arr_outside)

        arr_min.append(val_avg)
        arr_max.append(val_avg + val_std)
        arr_mean.append(val_avg - val_std)
    return arr_min, arr_max, arr_mean


def set_plot_inside(a, X, min, max, mean, color, label):
    a.fill_between(x=X, y1=max, y2=mean, color=color, alpha=0.2)
    a.plot(X, min, color=color, label=label)


def set_graph(X, data, ylable, pathFull, pathOri, save_name, extra_SARSA):
    f, a = plt.subplots()

    a.set_xlabel('Episodes', fontsize=20)
    a.set_ylabel(ylable, fontsize=25)

    # solving the special cases of 21,000 and original
    data_full = np.load(pathFull)
    data_ori = np.load(pathOri)
    data_S = np.load(extra_SARSA)
    arr_full = []
    arr_ori = []
    arr_S = []
    for i in range(actual_n):
        arr_outside_full = []
        arr_outside_ori = []
        arr_outside_S = []

        for j in range(smooth_factor):
            actual_index = i * smooth_factor + j

            avg_full = get_avg_each_ep(data_full, actual_index)
            arr_outside_full.append(avg_full)

            avg_ori = get_avg_each_ep(data_ori, actual_index)
            arr_outside_ori.append(avg_ori)

            avg_S = get_avg_each_ep(data_S, actual_index)
            arr_outside_S.append(avg_S)

        arr_full.append(np.mean(arr_outside_full))
        arr_ori.append(np.mean(arr_outside_ori))
        arr_S.append(np.mean(arr_outside_S))

    # solving other cases
    for i in range(n_gr):
        min, max, mean = get_results(data[i])
        set_plot_inside(a, X, min, max, mean, color_gr[i], type_test[i])

    a.plot(X, arr_full, color='red', label='21,000')
    a.plot(X, arr_ori, color='black', label='original')
    a.plot(X, arr_S, color='grey', label='SARSA - 0.2')

    a.legend()

    f.savefig(f'{save_name}.pdf')

#######################################
if __name__ == "__main__":
    X = range(0, EP, smooth_factor)

    set_graph(X, ha_count_Q, 'Hazard', './full/Q_ha.npy', './Q_ha.npy', 'ha_count_Q_combine', './extraSARSA/SARSA_ha.npy')
    set_graph(X, ha_count_SARSA, 'Hazard', './full/SARSA_ha.npy', './SARSA_ha.npy', 'ha_count_SARSA_combine', './extraSARSA/SARSA_ha.npy')
    set_graph(X, re_Q, 'Reward', './full/Q_re.npy', './Q_re.npy', 'reward_Q_combine', './extraSARSA/SARSA_re.npy')
    set_graph(X, re_SARSA, 'Reward', './full/SARSA_re.npy', './SARSA_re.npy', 'reward_SARSA_combine', './extraSARSA/SARSA_re.npy')

    plt.show()
