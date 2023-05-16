import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# ---------------------------- ハイパーパラメータ ----------------------------
LR_A = 0.0001  # Actorの学習率
LR_C = 0.0002  # Criticの学習率
GAMMA = 0.9  # 報酬の割引率
TAU = 0.01  # ソフトアップデート更新率
BATCH_SIZE = 32  # Experiment Replay Buffer バッチサイズ
OUTPUT_GRAPH = True  # Tensorflow計算グラフ
POLICY_DELAY = 2  # ポリシーネットワークの遅延更新
NOISE_CLIP = 0.5  # アクションノイズのクリッピング
STD_DEV_NOISE = 0.3  # アクションノイズの標準偏差


# ----------------------------------- TD3 ------------------------------------


class RL(object):

    def __init__(self, s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, a_bound):
        """
        Td3クラスを初期化する

        Parameters(パラメータ)
        ----------
        s_dim : int
            状態の次元数
        r_dim : int
            リソースの次元数 タスクの数
        b_dim : int
            帯域幅の次元数
        o_dim : int
            オフロードの次元数
        r_bound : float
            リソース値の上限
        b_bound : float
            帯域幅値の上限

        Attributes(属性)
        ----------
        memory_capacity : int
            経験再生バッファの容量
        s_dim : int
            状態の次元数
        a_dim : int
            行動の次元数
        r_dim : int
            リソースの次元数 タスクの数
        b_dim : int
            帯域幅の次元数
        o_dim : int
            オフロードの次元数
        r_bound : float
            リソース値の上限
        b_bound : float
            帯域幅値の上限
        S : tf.placeholder
            現在の状態を表すプレースホルダ
        S_ : tf.placeholder
            次の状態を表すプレースホルダ
        R : tf.placeholder
            報酬を表すプレースホルダ
        memory : np.array
            経験再生バッファ
        pointer : int
            経験再生バッファ内の現在の位置を示すポインタ
        sess : tf.Session
            計算を実行するためのTensorFlowセッション
        a : tf.Tensor
            Actorネットワークの出力
        aTrain : tf.Operation
            Actorネットワークの重みを更新するオペレーション
        cTrain : tf.Operation
            Criticネットワークの重みを更新するオペレーション
        """

        ''' 1. クラス変数を初期化 '''
        # 経験再生バッファ(エージェントの過去の状態、行動、報酬、次の状態の経験を保存するためのメモリ)の容量を設定
        self.memory_capacity = 10000

        # 各要素の範囲(次元数)を設定
        self.s_dim = s_dim
        self.a_dim = r_dim + b_dim + o_dim
        self.r_dim = r_dim
        self.b_dim = b_dim
        self.o_dim = o_dim

        # 計算資源と帯域幅の上限(bound)を設定
        self.r_bound = r_bound
        self.b_bound = b_bound
        self.a_bound = a_bound

        ''' 2. TensorFlowで計算グラフを構築する際に実際のデータが後から渡される変数(プレースホルダ)を初期化 '''
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        ''' 3. 経験再生バッファを初期化 '''
        # s_dim + a_dim + r + s_dim
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + self.a_dim + 1), dtype=np.float32)

        # 経験再生バッファ内の現在の位置を示すポインタを初期化
        self.pointer = 0

        ''' 4. TensorFlowのセッションを開始 '''
        self.sess = tf.Session()

        # 変数actor_update_counterの追加
        self.actor_update_counter = 0

        ''' 5. 現在の状態sに対する行動aと行動の価値qを計算 '''
        # Actorに現在の状態sを与え、最適な行動aを出力させる
        self.a = self.actor(self.S, )

        # Criticに現在の状態sと先程計算した行動aを与え、行動の価値qを出力
        q1, q2 = self.critic(self.S, self.a, )

        # ActorとCriticの学習可能なパラメータを取得(ステップ毎に目標パラメータを学習パラメータで置き換えるため)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        ''' 6. ターゲットネットワークの重みをソフトアップデートで更新するための指数移動平均オブジェクトを作成'''
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        ''' 7. ステップ毎に重みを更新する関数を定義し、次の状態s_における次の行動a_とQ値を計算'''

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # ステップ毎にターゲットネットワークの重み(a_params, c_params)を更新するソフトアップデート操作をリストに格納
        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation

        # 次の状態s_における次の行動a_とQ値を計算
        a_ = self.actor(self.S_, reuse=True, custom_getter=ema_getter)

        # ノイズ追加
        noise = tf.clip_by_value(tf.random.normal(tf.shape(a_), stddev=STD_DEV_NOISE), -NOISE_CLIP, NOISE_CLIP)
        a_ = tf.clip_by_value(a_ + noise, -self.a_bound, self.a_bound)

        q1_, q2_ = self.critic(self.S_, a_, reuse=True, custom_getter=ema_getter)

        ''' 8. ActorとCriticの損失関数を定義し、最適化アルゴリズムを使用してネットワークの重みを更新 '''
        # ここでは学習プロセスの定義が行われ、learn関数内では実際に学習プロセスが実行されている。

        # ---------------------------- Actorの学習 ----------------------------

        # Actorの損失関数
        a_loss = - tf.reduce_mean(q1)

        # 最適化アルゴリズムを使用して損失関数を最小化。Actorのパラメータ=重み(a_param)が更新される
        self.aTrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)  # 学習率LR_A

        # ---------------------------- Criticの学習 ----------------------------

        # ソフトアップデート操作(target_update)がcriticの学習と一緒に実行されるようにする
        with tf.control_dependencies(target_update):
            # 2つのターゲットCriticネットワーク（q1_およびq2_）の出力の最小値を使用

            # ターゲットQ値 (報酬Rと割引された次の状態のQ値の和)
            q_target = self.R + GAMMA * tf.minimum(q1_, q2_)

            # Criticの損失関数 (ターゲットQ値と予測Q値(q)の二乗誤差の平均)
            td_error1 = tf.losses.mean_squared_error(labels=q_target, predictions=q1)
            td_error2 = tf.losses.mean_squared_error(labels=q_target, predictions=q2)

            # ☆
            # 最適化アルゴリズムを使用して損失関数を最小化。Criticのパラメータ=重み(c_param)が更新される
            self.cTrain = tf.group(tf.train.AdamOptimizer(LR_C).minimize(td_error1, var_list=c_params),
                                   tf.train.AdamOptimizer(LR_C).minimize(td_error2, var_list=c_params))  # 学習率LR_C

        ''' 9. グローバル変数を初期化。TensorFlowの計算グラフを保存'''
        # TensorFlowセッション内の全ての変数を初期化
        self.sess.run(tf.global_variables_initializer())

        # ログディレクトリに計算グラフを保存
        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        """
        現在の状態sを入力として受け取り、Actorが推奨する行動を返す。

        Parameters
        ----------
        s : numpy.ndarray
            現在の状態を表す1次元のNumPy配列。

        Returns
        -------
        numpy.ndarray
            Actorが推奨する行動を表す1次元のNumPy配列。
        """
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        """
        経験再生バッファから経験(状態s、行動a、報酬r、次の状態s_)を選択し、そのデータでActorとCriticのパラメータ(重み)を更新する
        学習後、ターゲットネットワークのパラメータ(重み)はソフトアップデートで更新される

        Returns
        -------
        None
        """

        # 経験再生バッファからランダムにBATCH_SIZE分のindexを選択
        indices = np.random.choice(self.memory_capacity, size=BATCH_SIZE)  # Experience Replay (経験再生バッファからランダムに学習セットを選択)

        # 選択したindexに対応する経験(状態s、行動a、報酬r、次の状態s_)を経験再生バッファから取得
        bt = self.memory[indices, :]

        # btから状態s、行動a、報酬r、次の状態s_を抽出して、個別の配列に格納
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        #  学習の順番の変更(actor→criticからcritic→actorへ)
        # Criticの学習を実行
        # 状態s、行動a、報酬r、次の状態s_を入力として、ターゲットQ値と予測Q値の二乗誤差を最小化するように重みを更新する
        self.sess.run(self.cTrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        # 固定間隔(POLICY_DELAY)でActorネットワークとターゲットネットワークを更新
        if self.actor_update_counter % POLICY_DELAY == 0:
            # Actorの学習を実行。状態sを入力として、Q値を最大化するように行動aの勾配を更新する
            self.sess.run(self.aTrain, {self.S: bs})

        # Actorネットワークの更新カウンターをインクリメント
        self.actor_update_counter += 1

    def store_transition(self, s, a, r, s_):
        """
        エージェントの経験(現在の状態s、行動a、報酬r、次の状態s_)を経験再生バッファに保存する

        Parameters
        ----------
        s : np.ndarray
            現在の状態s
        a : np.ndarray
            その状態で選択された行動a
        r : float
            その行動によって得られる報酬r
        s_ : np.ndarray
            次の状態s_

        Returns
        -------
        None

        """

        # エージェントの経験(現在の状態s、行動a、報酬r、次の状態s_)を1つの配列に水平方向に連結したデータ
        transition = np.hstack((s, a, [r], s_))

        # 経験再生バッファに保存するindexは、ポインタを経験再生バッファの容量で割った余り(循環構造)
        # これにより、経験再生バッファが一杯になった場合、古いデータから上書きされるようになる
        index = self.pointer % self.memory_capacity

        # 遷移データを保存
        self.memory[index, :] = transition

        # ポインタを1つ進める
        self.pointer += 1

    def actor(self, s, reuse=None, custom_getter=None):
        """
        状態sを入力として受け取り、行動aを出力するActor

        Parameters
        ----------
        s : tf.Tensor
            入力状態
        reuse : bool, optional
            変数を再利用するかどうか。デフォルトはNone。
        custom_getter : callable, optional
            カスタム変数の取得方法。デフォルトはNone。

        Returns
        -------
        a : tf.Tensor
            Actorの出力 (行動a)
        """

        # actor関数の引数reuseがNoneならトレーニング可能(True)、それ以外ならトレーニング不可能(false)
        trainable = True if reuse is None else False

        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            n_l = 50  # 中間層のニューロン数

            # 隠れ層l1 入力の状態sを受け取り、全結合層を適用
            net = tf.layers.dense(s, n_l, activation=tf.nn.relu, name='l1', trainable=trainable)

            # リソース推定のための5つの全結合層を定義。入力:net, 出力:self.r_dim
            layer_r0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='r_0', trainable=trainable)
            layer_r1 = tf.layers.dense(layer_r0, n_l, activation=tf.nn.relu, name='r_1', trainable=trainable)
            layer_r2 = tf.layers.dense(layer_r1, n_l, activation=tf.nn.relu, name='r_2', trainable=trainable)
            layer_r3 = tf.layers.dense(layer_r2, n_l, activation=tf.nn.relu, name='r_3', trainable=trainable)
            layer_r4 = tf.layers.dense(layer_r3, self.r_dim, activation=tf.nn.relu, name='r_4', trainable=trainable)

            # ★追加
            layer_r4 = tf.clip_by_value(layer_r4, 0, self.r_bound)  # r_boundでクリップ

            # バンド幅推定のための5つの全結合層を定義。入力:net, 出力:self.b_dim
            layer_b0 = tf.layers.dense(net, n_l, activation=tf.nn.relu, name='b_0', trainable=trainable)
            layer_b1 = tf.layers.dense(layer_b0, n_l, activation=tf.nn.relu, name='b_1', trainable=trainable)
            layer_b2 = tf.layers.dense(layer_b1, n_l, activation=tf.nn.relu, name='b_2', trainable=trainable)
            layer_b3 = tf.layers.dense(layer_b2, n_l, activation=tf.nn.relu, name='b_3', trainable=trainable)
            layer_b4 = tf.layers.dense(layer_b3, self.b_dim, activation=tf.nn.relu, name='b_4', trainable=trainable)

            # ★追加
            layer_b4 = tf.clip_by_value(layer_b4, 0, self.b_bound)  # b_boundでクリップ

            # オフローディング確率(確率:0 - 1)を推定する
            # layerリストを初期化。タスクごとに4つの全結合層の出力を格納する
            layer = [["layer" + str(task_id) + str(layer) for layer in range(4)] for task_id in range(self.r_dim)]

            # 各全結合層の名前を格納するリスト(実質layerと同じ)
            name = [["layer" + str(task_id) + str(layer) for layer in range(4)] for task_id in range(self.r_dim)]

            # オフローディングの確率を格納するリスト
            task = ["task" + str(task_id) for task_id in range(self.r_dim)]

            # ソフトマックス関数の名前を格納するリスト
            softmax = ["softmax" + str(task_id) for task_id in range(self.r_dim)]

            # 各タスクに対してオフローディング確率を計算
            for task_id in range(self.r_dim):
                # 4つの全結合層を連続して適用し、オフローディングの確率を計算するための基礎となる値を求める
                # 入力:net, 出力:self.o_dim / self.r_dim (各タスクのオフローディング確率の次元数)
                layer[task_id][0] = tf.layers.dense(net, n_l, activation=tf.nn.relu,
                                                    name=name[task_id][0], trainable=trainable)
                layer[task_id][1] = tf.layers.dense(layer[task_id][0], n_l, activation=tf.nn.relu,
                                                    name=name[task_id][1], trainable=trainable)
                layer[task_id][2] = tf.layers.dense(layer[task_id][1], n_l, activation=tf.nn.relu,
                                                    name=name[task_id][2], trainable=trainable)
                layer[task_id][3] = tf.layers.dense(layer[task_id][2], (self.o_dim / self.r_dim), activation=tf.nn.relu,
                                                    name=name[task_id][3], trainable=trainable)

                # 最後の全結合層の出力にソフトマックス関数を適用(確率値の合計が1になるように正規化)し、オフローディング確率を計算
                task[task_id] = tf.nn.softmax(layer[task_id][3], name=softmax[task_id])

            # layer_r4(リソースの次元)とlayer_b4(帯域幅の次元)を結合して、最終的な行動aの初期値を作成
            a = tf.concat([layer_r4, layer_b4], 1)

            # 各タスクのオフローディング確率(task[task_id])を行動aに結合していく
            for task_id in range(self.r_dim):
                a = tf.concat([a, task[task_id]], 1)

            return a

    def critic(self, s, a, reuse=None, custom_getter=None):
        """
        状態sと行動aを入力として受け取り、2つの行動の価値Q値（q1, q2）を出力するCritic

        Parameters
        ----------
        s : tf.Tensor
            入力状態s
        a : tf.Tensor
            行動a
        reuse : bool, optional
            ネットワークの重みを再利用するかどうか。デフォルトはNone。
        custom_getter : callable, optional
            カスタムゲッター関数。デフォルトはNone。

        Returns
        -------
        q1 : tf.Tensor
            Criticの出力 最初のQ値(行動の価値)
        q2 : tf.Tensor
            Criticの出力 2番目のQ値(行動の価値)
        """

        # critic関数の引数reuseがNoneならトレーニング可能(True)、それ以外ならトレーニング不可能(false)
        trainable = True if reuse is None else False

        # (Q value範囲：0 - inf)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            # 各中間層のニューロン数を設定
            n_l = 50

            # 双子のクリティック（Twin Critic）

            # ---------------------------- Critic 1 ----------------------------

            # 状態sと行動aを結合するための重みとバイアスを定義
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l], trainable=trainable)  # 状態sに対する重み行列を初期化
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l], trainable=trainable)  # 行動aに対する重み行列を初期化
            b1 = tf.get_variable('b1', [1, n_l], trainable=trainable)  # 第1層のバイアス項を初期化

            # 状態sと行動aを結合(入力s,aと重みの行列積の和にバイアスを加えている)し、ReLU活性化関数を適用
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # 3つの隠れ層を追加し、それぞれに活性化関数ReLUを適用
            net_2 = tf.layers.dense(net_1, n_l, activation=tf.nn.relu, trainable=trainable)
            net_3 = tf.layers.dense(net_2, n_l, activation=tf.nn.relu, trainable=trainable)
            net_4 = tf.layers.dense(net_3, n_l, activation=tf.nn.relu, trainable=trainable)

            # 最後に全結合層を追加し、活性化関数ReLUを適用してQ値を求める
            q1 = tf.layers.dense(net_4, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)

            # ---------------------------- Critic 2 ----------------------------
            w2_s = tf.get_variable('w2_s', [self.s_dim, n_l], trainable=trainable)
            w2_a = tf.get_variable('w2_a', [self.a_dim, n_l], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l], trainable=trainable)

            net_5 = tf.nn.relu(tf.matmul(s, w2_s) + tf.matmul(a, w2_a) + b2)

            net_6 = tf.layers.dense(net_5, n_l, activation=tf.nn.relu, trainable=trainable)
            net_7 = tf.layers.dense(net_6, n_l, activation=tf.nn.relu, trainable=trainable)
            net_8 = tf.layers.dense(net_7, n_l, activation=tf.nn.relu, trainable=trainable)

            q2 = tf.layers.dense(net_8, 1, activation=tf.nn.relu, trainable=trainable)  # Q(s,a)

            # 2つのQ値を出力
            return q1, q2

    @staticmethod
    def get_inf():
        """
        ハイパーパラメータのゲッター
        """
        return LR_A, LR_C, GAMMA, TAU, BATCH_SIZE, OUTPUT_GRAPH, POLICY_DELAY, NOISE_CLIP, STD_DEV_NOISE
