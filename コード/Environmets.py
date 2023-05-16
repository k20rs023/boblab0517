import random
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from render import Demo

#####################  ハイパーパラメータ  ####################
KSU_LOCATION = "KSU"
KSU0_LOCATION = "KSU0"
TASK_NUM = 90
EDGE_NUM = 10
LIMIT = 10
MAX_EP_STEPS = 3000
TXT_NUM = 92
r_bound = 1e9 * 0.063
b_bound = 1e9
CPU = 100
Memory = 100
HDD = 100

#####################  function  ####################

def trans_rate(task_loc, edge_loc):
    B = 2e6
    P = 0.25
    d = np.sqrt(np.sum(np.square(task_loc[0] - edge_loc))) + 0.01
    h = 4.11 * math.pow(3e8 / (4 * math.pi * 915e6 * d), 2)
    N = 1e-10
    return B * math.log2(1 + P * h / N)

def BandwidthTable(edge_num):
    BandwidthTable = np.zeros((edge_num, edge_num))
    for i in range(0, edge_num):
        for j in range(i+1, edge_num):
                BandwidthTable[i][j] = 1e9
    return BandwidthTable

def two_to_one(two_table):
    one_table = two_table.flatten()
    return one_table

def generate_state(two_table, T, E, x_min, y_min):
    # 初期化
    one_table = two_to_one(two_table)
    S = np.zeros((len(E) + one_table.size + len(T) + len(T)*2))
    # transform
    count = 0
    # オフロード先エッジサーバの利用可能な計算資源を取得
    for edge in E:
        S[count] = edge.capability/(r_bound*10)
        count += 1
    # 各接続の利用可能な帯域幅を取得
    for i in range(len(one_table)):
        S[count] = one_table[i]/(b_bound*10)
        count += 1
    # それぞれのタスクのオフロード状況を取得（0 or 1）
    for task in T:
        S[count] = task.req.edge_id/100
        count += 1
    # オフロードするエッジサーバの位置情報を取得
    for task in T:
        S[count] = (task.loc[0][0] + abs(x_min))/1e5
        S[count+1] = (task.loc[0][1] + abs(y_min))/1e5
        count += 2
    return S

def generate_action(R, B, O):
    # 計算資源割り当て（連続値）
    a = np.zeros(TASK_NUM + TASK_NUM + EDGE_NUM * TASK_NUM)
    a[:TASK_NUM] = R / r_bound
    # 帯域幅割り当て（連続値）
    a[TASK_NUM:TASK_NUM + TASK_NUM] = B / b_bound
    # オフロード決定（離散値）
    base = TASK_NUM + TASK_NUM
    for task_id in range(TASK_NUM):
        a[base + int(O[task_id])] = 1
        base += EDGE_NUM
    return a

# タスクデータ取得
def get_minimum():
    cal = np.zeros((1, 2))
    for data_num in range(TXT_NUM):
        data_name = str("%03d" % (data_num + 1))  # plus zero
        file_name = KSU0_LOCATION + "_30sec_" + data_name + ".txt"
        file_path = "data/" + KSU0_LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num
        line_num = 0
        for line in f1:
            line_num += 1
        # .txtからデータを収集
        data = np.zeros((line_num, 2))
        index = 0
        for line in f1:
            data[index][0] = line.split()[1]  # x
            data[index][1] = line.split()[2]  # y
            index += 1
        # データをcalに格納
        cal = np.vstack((cal, data))
    return min(cal[:, 0]), min(cal[:, 1])

# エッジサーバの位置情報取得
def proper_edge_loc(edge_num):

    # e_lの初期化
    e_l = np.zeros((edge_num, 2))
    # データの平均を計算
    group_num = math.floor(TXT_NUM / edge_num)
    edge_id = 0
    for base in range(0, group_num*edge_num, group_num):
        for data_num in range(base, base + group_num):
            data_name = str("%03d" % (data_num + 1))  # plus zero
            file_name = KSU_LOCATION + "_30sec_" + data_name + ".txt"
            file_path = "data/" + KSU_LOCATION + "/" + file_name
            f = open(file_path, "r")
            f1 = f.readlines()
            # line_numを取得しデータを初期化
            line_num = 0
            for line in f1:
                line_num += 1
            data = np.zeros((line_num, 2))
            # .txtからデータを収集
            index = 0
            for line in f1:
                data[index][0] = line.split()[1]  # x
                data[index][1] = line.split()[2]  # y
                index += 1
            # 収集したデータをスタックとして積み重ねて保存
            if data_num % group_num == 0:
                cal = data
            else:
                cal = np.vstack((cal, data))
        e_l[edge_id] = np.mean(cal, axis=0)
        edge_id += 1
    return e_l

##########################  Offload Edge Server  ###########################
class OES():
    def __init__(self, task_id, data_num):
        self.task_id = task_id  # number of the task
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # num_stepを計算しself.mobを定義
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = KSU0_LOCATION + "_30sec_" + data_num + ".txt"
        file_path = "data/" + KSU0_LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))

        # self.mobにデータを書き込む
        now_sec = 0
        for line in f1:
            for sec in range(30):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30
        self.loc[0] = self.mob[0]

    def generate_request(self, edge_id):
        self.req = Request(self.task_id, edge_id)

    def request_update(self):
        # default request.state == [5：disconnection（切断状態）,6：migration（移動状態）]
        if self.req.state == 5:
            self.req.timer += 1
        else:
            self.req.timer = 0
            if self.req.state == 0:
                self.req.state = 1
                self.req.u2e_size = self.req.tasktype.req_u2e_size
                self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
            elif self.req.state == 1:
                if self.req.u2e_size > 0:
                    self.req.u2e_size -= trans_rate(self.loc, self.req.edge_loc)
                else:
                    self.req.state = 2
                    self.req.process_size = self.req.tasktype.process_loading
                    self.req.process_size -= self.req.resource
            elif self.req.state == 2:
                if self.req.process_size > 0:
                    self.req.process_size -= self.req.resource
                else:
                    self.req.state = 3
                    self.req.e2u_size = self.req.tasktype.req_e2u_size
                    self.req.e2u_size -= 10000  # value is small,so simplify
            else:
                if self.req.e2u_size > 0:
                    self.req.e2u_size -= 10000  # B*math.log(1+SINR(self.task.loc, self.offloading_serv.loc), 2)/(8*time_scale)
                else:
                    self.req.state = 4

    def task_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]   # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf

class Request():
    def __init__(self, task_id, edge_id):
        # id
        self.task_id = task_id
        self.edge_id = edge_id
        self.edge_loc = 0
        # state（状態）
        self.state = 5     # 5: not connect
        self.pre_state=5
        # transmission size（送信サイズ）
        self.u2e_size = 0
        self.process_size = 0
        self.e2u_size = 0
        # edge state（エッジサーバの状態）
        self.resource = 0
        self.mig_size = 0
        # タスクタイプ
        self.tasktype = TaskType()
        self.last_offlaoding = 0
        # timer
        self.timer = 0

class TaskType():
    def __init__(self):
        ##Objection detection: VOC SSD300
        # transmission
        self.req_u2e_size = 300 * 300 * 3 * 1
        self.process_loading = 300 * 300 * 3 * 4
        self.req_e2u_size = 4 * 4 + 20 * 4
        # migration
        self.migration_size = 2e9
    def task_inf(self):
        return "req_u2e_size:" + str(self.req_u2e_size) + "\nprocess_loading:" + str(self.process_loading) + "\nreq_e2u_size:" + str(self.req_e2u_size)

##################  Other Edge Servers  ###################
class ES():
    def __init__(self, edge_id, loc, cpu, mem, hdd):
        self.edge_id = edge_id  # edge server number
        self.loc = loc
        self.capability = 1e9 * 0.063
        self.task_group = []
        self.limit = LIMIT
        self.connection_num = 0
        self.cpu = cpu
        self.mem = mem
        self.hdd = hdd
    def maintain_request(self, R, T):
        for task in T:
            # 接続タスク数
            self.connection_num = 0
            for task_id in self.task_group:
                if T[task_id].req.state != 6:
                    self.connection_num += 1
            # リクエストの維持
            if task.req.edge_id == self.edge_id and self.capability - R[task.task_id] > 0:
                # 予備接続の維持
                if task.req.task_id not in self.task_group and self.connection_num+1 <= self.limit:
                    # 初回：どのエッジサーバにも属さない（task_group）
                    self.task_group.append(task.task_id)  # task_groupに追加
                    task.req.state = 0  # 接続準備
                    # リクエストの通知
                    task.req.edge_id = self.edge_id
                    task.req.edge_loc = self.loc

                # 計算資源の派遣（リソース提供）
                task.req.resource = R[task.task_id]
                self.capability -= R[task.task_id]

    def migration_update(self, O, B, table, T, E):

        # マイグレーションを維持する
        for task_id in self.task_group:
            # マイグレーションを準備する
            if T[task_id].req.edge_id != O[task_id]:
                # 初期化
                ini_edge = int(T[task_id].req.edge_id)
                target_edge = int(O[task_id])
                if table[ini_edge][target_edge] - B[task_id] >= 0:
                    # マイグレーション中だが，別のエッジサーバにオフロード(step 1)
                    if T[task_id].req.state == 6 and target_edge != T[task_id].req.last_offlaoding:
                        # 帯域幅の削減
                        table[ini_edge][target_edge] -= B[task_id]
                        # マイグレーション開始
                        T[task_id].req.mig_size = T[task_id].req.tasktype.migration_size
                        T[task_id].req.mig_size -= B[task_id]
                        #print("task", T[task_id].req.task_id, ":migration step 1")
                    # first try to migration(step 1)
                    elif T[task_id].req.state != 6:
                        table[ini_edge][target_edge] -= B[task_id]
                        # マイグレーション開始
                        T[task_id].req.mig_size = T[task_id].req.tasktype.migration_size
                        T[task_id].req.mig_size -= B[task_id]
                        # 直前の状態を保存
                        T[task_id].req.pre_state = T[task_id].req.state
                        # マイグレーション中に旧エッジサーバとの接続を解除する
                        T[task_id].req.state = 6
                        #print("task", T[task_id].req.task_id, ":migration step 1")
                    elif T[task_id].req.state == 6 and target_edge == T[task_id].req.last_offlaoding:
                        # マイグレーションの継続(step 2)
                        if T[task_id].req.mig_size > 0:
                            # reduce the bandwidth
                            table[ini_edge][target_edge] -= B[task_id]
                            T[task_id].req.mig_size -= B[task_id]
                            #print("task", T[task_id].req.task_id, ":migration step 2")
                        # マイグレーションの終了(step 3)
                        else:
                            # 接続タスク数
                            target_connection_num = 0
                            for target_task_id in E[target_edge].task_group:
                                if T[target_task_id].req.state != 6:
                                    target_connection_num += 1
                            #print("task", T[task_id].req.task_id, ":migration step 3")
                            # 別のエッジサーバに変更
                            if E[target_edge].capability - T[task_id].req.resource >= 0 and target_connection_num + 1 <= E[target_edge].limit:
                                # 新しいエッジサーバに登録
                                E[target_edge].capability -= T[task_id].req.resource
                                E[target_edge].task_group.append(task_id)
                                self.task_group.remove(task_id)
                                # リクエストをアップデート
                                # id
                                T[task_id].req.edge_id = E[target_edge].edge_id
                                T[task_id].req.edge_loc = E[target_edge].loc
                                # 直前の状態をリリースし送信処理を継続する
                                T[task_id].req.state = T[task_id].req.pre_state
                                #print("task", T[task_id].req.task_id, ":migration finish")
            # 直前のオフロード決定を保存
            T[task_id].req.last_offlaoding = int(O[task_id])

        return table

    # 全ての計算資源をリリース
    def release(self):
        self.capability = 1e9 * 0.063

####################  Policy  #######################

# 優先するポリシーの定義
class priority_policy():
    # 優先順位を生成
    def generate_priority(self, T, E, priority):
        for task in T:
            # オフロード優先順位リストを取得
            dist = np.zeros(EDGE_NUM)
            for edge in E:
                edge.cpu <= 50
                if edge.cpu > 50:
                    edge.mem <= 50
                    if edge.cpu > 50 & edge.mem > 50:
                        edge.hdd <= 50
                        if edge.cpu > 50 & edge.mem > 50 & edge.hdd > 50:
                            dist[edge.edge_id] = np.sqrt(np.sum(np.square(task.loc[0] - edge.loc)))
            dist_sort = np.sort(dist)
            for index in range(EDGE_NUM):
                priority[task.task_id][index] = np.argwhere(dist == dist_sort[index])[0]
        return priority

    # 優先順位に応じたオフロード先エッジの決定
    def indicate_edge(self, O, T, priority):
        edge_limit = np.ones((EDGE_NUM)) * LIMIT
        for task in T:
            for index in range(EDGE_NUM):
                if edge_limit[int(priority[task.task_id][index])] - 1 >= 0:
                    edge_limit[int(priority[task.task_id][index])] -= 1
                    O[task.task_id] = priority[task.task_id][index]
                    break
        return O

    def resource_update(self, R, E ,T):
        for edge in E:
            # 接続タスク数のカウント
            connect_num = 0
            for task_id in edge.task_group:
                if T[task_id].req.state != 5 and T[task_id].req.state != 6:
                    connect_num += 1
            for task_id in edge.task_group:
                # リクエスト状態が5or6（切断状態）で，計算資源を提供する必要がない場合
                if T[task_id].req.state == 5 or T[task_id].req.state == 6:
                    R[task_id] = 0
                else:
                    R[task_id] = edge.capability/(connect_num+2)  # reserve the resource to those want to migration
        return R

    def bandwidth_update(self, O, table, B, T, E):
        for task in T:
            share_number = 1
            ini_edge = int(task.req.edge_id)
            target_edge = int(O[task.req.task_id])
            # マイグレーション不要
            if ini_edge == target_edge:
                B[task.req.task_id] = 0
            # マイグレーションのための帯域を提供
            else:
                # マイグレーション先のエッジサーバのタスクで帯域を共有
                for task_id in E[target_edge].task_group:
                    if O[task_id] == ini_edge:
                        share_number += 1
                # 旧エッジサーバから移行先エッジサーバまでのタスクと帯域を共有
                for ini_task_id in E[ini_edge].task_group:
                    if ini_task_id != task.req.task_id and O[ini_task_id] == target_edge:
                        share_number += 1
                # 帯域の割り当て
                B[task.req.task_id] = table[min(ini_edge, target_edge)][max(ini_edge, target_edge)] / (share_number+2)

        return B

####################  Environment  ######################

class Environment():
    def __init__(self):
        self.step = 30
        self.time = 0
        self.edge_num = EDGE_NUM  # エッジサーバ数
        self.task_num = TASK_NUM  # タスク数
        # 環境オブジェクトの定義
        self.reward_all = []
        self.T = []
        self.fin_req_count = 0
        self.prev_count = 0
        self.rewards = 0
        self.R = np.zeros((self.task_num))
        self.O = np.zeros((self.task_num))
        self.B = np.zeros((self.task_num))
        self.table = BandwidthTable(self.edge_num)
        self.priority = np.zeros((self.task_num, self.edge_num))
        self. E = []
        self.x_min, self.y_min = get_minimum()

        self.e_l = 0
        self.model = 0

    def get_inf(self):
        # s_dim
        self.reset()
        s = generate_state(self.table, self.T, self.E, self.x_min, self.y_min)
        s_dim = s.size

        # a_dim
        r_dim = len(self.T)
        b_dim = len(self.T)
        o_dim = self.edge_num * len(self.T)

        # 計算資源の最大値
        r_bound = self.E[0].capability

        # 帯域幅の最大値
        b_bound = self.table[0][1]
        b_bound = b_bound.astype(np.float32)

        # a_bound追加
        a_bound = r_bound + b_bound

        # タスクサイズ
        task = TaskType()
        task_inf = task.task_inf()

        return s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, a_bound, task_inf, LIMIT, KSU_LOCATION, KSU0_LOCATION, CPU, Memory, HDD

    def reset(self):
        # reset time
        self.time = 0
        # reward（報酬）
        self.reward_all = []
        # task
        self.T = []
        self.fin_req_count = 0
        self.prev_count = 0
        data_num = random.sample(list(range(TXT_NUM)), self.task_num)
        for i in range(self.task_num):
            new_task = OES(i, data_num[i])
            self.T.append(new_task)
        # Resource
        self.R = np.zeros((self.task_num))
        # Offlaoding
        self.O = np.zeros((self.task_num))
        # bandwidth
        self.B = np.zeros((self.task_num))
        # bandwidth table
        self.table = BandwidthTable(self.edge_num)
        # server
        self.E = []
        e_l = proper_edge_loc(self.edge_num)
        #CPU使用率をランダムに生成
        e_cpu = random.sample(list(range(CPU)), self.edge_num)
        #メモリ使用率をランダムに生成
        e_mem = random.sample(list(range(Memory)), self.edge_num)
        #HDD使用率をランダムに生成
        e_hdd = random.sample(list(range(HDD)), self.edge_num)
        for i in range(self.edge_num):
            new_e = ES(i, e_l[i, :], e_cpu[i], e_mem[i], e_hdd[i])
            self.E.append(new_e)


            """
            print("edge", new_e.edge_id, "'s loc:\n", new_e.loc)
        print("========================================================")
        """
        # model
        self.model = priority_policy()

        # リクエストの初期化
        self.priority = self.model.generate_priority(self.T, self.E, self.priority)
        self.O = self.model.indicate_edge(self.O, self.T, self.priority)
        for task in self.T:
            task.generate_request(self.O[task.task_id])
        return generate_state(self.table, self.T, self.E, self.x_min, self.y_min)

    def ddpg_step_forward(self, a, r_dim, b_dim):
        # release the bandwidth
        self.table = BandwidthTable(self.edge_num)
        # release the resource
        for edge in self.E:
            edge.release()

        # 1秒ごとにポリシーを更新
        # resource update
        self.R = a[:r_dim]
        # bandwidth update
        self.B = a[r_dim:r_dim + b_dim]
        # offloading update
        base = r_dim + b_dim
        for task_id in range(self.task_num):
            prob_weights = a[base:base + self.edge_num]

            # 以下のコードを追加
            prob_weights = np.maximum(prob_weights, 0)  # Ensure all values are non-negative
            sum_prob_weights = np.sum(prob_weights)
            if sum_prob_weights > 0:
                prob_weights /= sum_prob_weights  # Normalize the probabilities
            else:
                prob_weights = np.ones_like(prob_weights) / len(
                    prob_weights)  # Assign equal probability if all weights are zero

            # print("task", task_id, ":", prob_weights)
            action = np.random.choice(range(len(prob_weights)), p=prob_weights.ravel())  # select action w.r.t the actions prob
            base += self.edge_num
            self.O[task_id] = action


        # request update
        for task in self.T:
            # リクエストの状態を更新
            task.request_update()
            if task.req.timer >= 5:
                task.generate_request(self.O[task.task_id])  # 優先順位に従ってオフロード
            # 既にリクエストが終了している場合
            if task.req.state == 4:
             # rewards（報酬）
                self.fin_req_count += 1
               #最適なEdge Serverへオフロードできた場合の追加報酬
                c = [edge.cpu]
                m = [edge.mem]
                h = [edge.hdd]
                if min(c):
                    self.fin_req_count += 1
                elif edge.cpu > 50:
                    if min(m):
                        self.fin_req_count += 1
                    elif edge.cpu > 50 & edge.mem > 50:
                        if min(h):
                            self.fin_req_count += 1
                        elif edge.cpu > 50 & edge.mem > 50 & edge.hdd > 50:
                            self.fin_req_count += 0
                task.req.state = 5  # リクエストを"disconnect（切断）"にする
                self.E[int(task.req.edge_id)].task_group.remove(task.req.task_id)
                task.generate_request(self.O[task.task_id])  # 優先順位に従ってオフロード

        # edge update
        for edge in self.E:
            #1step終えたらCPU・メモリ・HDDの使用率をリセット
            if task.req.state == 5:
                edge.cpu = 0
                edge.mem = 0
                edge.hdd = 0
            #次のstepのCPU・メモリ・HDDの使用率をランダムに生成
                e_cpu = random.sample(list(range(CPU)), self.edge_num)
                e_mem = random.sample(list(range(Memory)), self.edge_num)
                e_hdd = random.sample(list(range(HDD)), self.edge_num)
                for i in range(self.edge_num):
                    edge.cpu = e_cpu[i]
                    edge.mem = e_mem[i]
                    edge.hdd = e_hdd[i]

            edge.maintain_request(self.R, self.T)
            self.table = edge.migration_update(self.O, self.B, self.table, self.T, self.E)

        # rewards
        self.rewards = self.fin_req_count - self.prev_count
        self.prev_count = self.fin_req_count

        if self.time % self.step == 0:
            for task in self.T:
                task.task_update(self.time)

        # update time
        self.time += 1

        # return s_, r
        return generate_state(self.table, self.T, self.E, self.x_min, self.y_min), self.rewards

    def text_render(self):
        print("R:", self.R)
        print("B:", self.B)
        print("O:", self.O)
        print("-------------------Offload Edge Server---------------------")
        for task in self.T:
            print("Offload Edge Server's loc:", task.loc)
            print("task", task.task_id)
            print("request state:", task.req.state)
            print("edge serve:", task.req.edge_id)
        print("-------------------Other Edge Servers---------------------")
        for edge in self.E:
            print("edge", edge.edge_id)
            print("location:", edge.loc)
            print("CPU使用率:", edge.cpu, "%")
            print("メモリ使用率:", edge.mem, "%")
            print("HDD使用率:", edge.hdd, "%")
            print("task_group:", edge.task_group)
        print("reward:", self.rewards)
        print("=====================  update  ===========================")

    def initial_screen_demo(self):
        self.canvas = Demo(self.E, self.T, self.O, MAX_EP_STEPS)

    def screen_demo(self):
        self.canvas.draw(self.E, self.T, self.O)