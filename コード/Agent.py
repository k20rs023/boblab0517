from Environmets import Environment
from RL import RL
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ---------------------------- ハイパーパラメータ ----------------------------
RL_NAME = 'td3'            # 使用しているアルゴリズム名(ddpg/td3)
CHECK_EPISODE = 4          # 探索パラメータの更新を判断するために使われるエピソード数
LEARNING_MAX_EPISODE = 30  # エージェントが学習を行う最大エピソード数
MAX_EP_STEPS = 3000        # 各エピソード内で実行される最大ステップ数
TEXT_RENDER = True         # テキストレンダリングを有効にするか
SCREEN_RENDER = False      # スクリーンレンダリングを有効にするか
CHANGE = True              # 探索パラメータを更新するか
SLEEP_TIME = 0.01          # 各ステップ間でプログラムを一時停止する時間


# -------------------------------- function -----------------------------------
def exploration(a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a


# -------------------------------- training -----------------------------------

if __name__ == "__main__":
    env = Environment()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, a_bound, task_inf, limit, edge_location, edge0_location, cpu, mem, hdd = env.get_inf()
    rl = RL(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, a_bound)
    LR_A, LR_C, GAMMA, TAU, BATCH_SIZE, OUTPUT_GRAPH, POLICY_DELAY, NOISE_CLIP, STD_DEV_NOISE = rl.get_inf()

    r_var = 1  # control exploration
    b_var = 1
    ep_reward = []
    # ep_utilization = []
    ep_a_loss = []  # Actorの損失
    ep_td_error = []  # TD_error
    r_v, b_v = [], []
    var_reward = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []
    epoch = 0
    t1 = time.time()
    while var_counter < LEARNING_MAX_EPISODE:
        # 初期化
        s = env.reset()
        ep_reward.append(0)

        if SCREEN_RENDER:
            env.initial_screen_demo()

        for j in range(MAX_EP_STEPS):
            time.sleep(SLEEP_TIME)  # コンソールを一時停止して見やすくする
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:  # コンソールにステップ毎の進捗を表示
                env.text_render()

            # rl
            # 状態に応じて行動を選択
            a = rl.choose_action(s)  # a = [R B O]
            # 探索のための行動選択にランダム性を持たせる
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # 遷移パラメータを格納
            s_, r = env.ddpg_step_forward(a, r_dim, b_dim)
            rl.store_transition(s, a, r / 10, s_)
            # 学習
            if rl.pointer == rl.memory_capacity:
                print("start learning")
            if rl.pointer > rl.memory_capacity:
                rl.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # 状態を置き換える（更新）
            s = s_
            # 報酬を集計
            ep_reward[episode] += r

            # 最後のエピソード
            if j == MAX_EP_STEPS - 1:
                var_reward.append(ep_reward[episode])
                r_v.append(r_var)
                b_v.append(b_var)

                print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], '###  r_var: %.2f ' % r_var,
                      'b_var: %.2f ' % b_var)
                string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[
                    episode] + '###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var

                epoch_inf.append(string)
                # variation change
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1
        # エピソードの終了
        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        episode += 1

    print('Running time: ', time.time() - t1)

    # ディレクトリ作成
    # dt_now = datetime.datetime(2022,1,7,19,00)
    dir_name = 'output/' + RL_NAME + '_' + str(LR_A) + 'a' + str(LR_C) + 'c' + str(NOISE_CLIP) + 'nc' + str(STD_DEV_NOISE) + 'sdn' + str(limit) + 'l' + str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:]))
    # ↑ str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' を削除
    if os.path.isdir(dir_name):
        os.rmdir(dir_name)
    os.makedirs(dir_name)
    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i + 1 for i in range(episode)], ep_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')

    # plot the CPU,Memory,HDD utilization rate
    """fig_utilization = plt.figure()
    plt.plot([i+1 for i in range(episode)], ep_utilization)
    plt.xlabel("episode")
    plt.ylabel("utilization rewards")
    fig_utilization.savefig(dir_name + '/utilization rewards.png')"""

    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')

    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('task_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')

    # 最後の LEARNING_MAX_EPISODE 個のエピソード報酬の平均値
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE, " epochs:",
          str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" + str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    # 最後の LEARNING_MAX_EPISODE 個のエピソード報酬の標準偏差 = データのばらつき
    print("the standard deviation of the rewards:", str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" + str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    # 最後の LEARNING_MAX_EPISODE 個のエピソード報酬の範囲（最大値と最小値の差） = 学習後のエピソード報酬の広がり
    print("the range of the rewards:",
          str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" + str(
        max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    f.close()
