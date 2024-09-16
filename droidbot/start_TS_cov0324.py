# helper file of droidbot
# it parses command arguments and send the options to droidbotX
import argparse
import input_manager
import input_policy
import env_manager
from droidbot import DroidBot
from droidmaster import DroidMaster
import droidbot_env
import numpy as np
import pickle
import time
from input_event import KeyEvent, TouchEvent, LongTouchEvent, ScrollEvent
import json
import matplotlib.pyplot as plt

import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines3 import A2C
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines3 import DQN
import coverage
# 启动覆盖率测量
cov = coverage.Coverage()
cov.start()

n_steps = 0 #used for saving model with callback

# save RL model in progress
def callback(_locals, _globals, save_every=1000):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps
    # Print stats every 1000 calls
    if (n_steps + 1) % save_every == 0:
        temp_int = int(time.time())
        print("saving while training model")
        _locals['self'].save('in_progress_model_{}.pkl'.format(temp_int))
    n_steps += 1
    return True


def parse_args():
    """
    parse command line input
    generate options including host name, port number
    """
    parser = argparse.ArgumentParser(description="Start DroidBot to test an Android app.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", action="store", dest="device_serial", required=False,
                        help="The serial number of target device (use `adb devices` to find)")
    parser.add_argument("-a", action="store", dest="apk_path", required=True,
                        help="The file path to target APK")
    parser.add_argument("-o", action="store", dest="output_dir",
                        help="directory of output")
    parser.add_argument("-policy", action="store", dest="input_policy", default=input_manager.DEFAULT_POLICY,
                        help='Policy to use for test input generation. '
                             'Default: %s.\nSupported policies:\n' % input_manager.DEFAULT_POLICY +
                             '  \"%s\" -- No event will be sent, user should interact manually with device; \n'
                             '  \"%s\" -- Use "adb shell monkey" to send events; \n'
                             '  \"%s\" -- Explore UI using a naive depth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a greedy depth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a naive breadth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a greedy breadth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a gym strategy;\n'
                             %
                             (
                                 input_policy.POLICY_NONE,
                                 input_policy.POLICY_MONKEY,
                                 input_policy.POLICY_NAIVE_DFS,
                                 input_policy.POLICY_GREEDY_DFS,
                                 input_policy.POLICY_NAIVE_BFS,
                                 input_policy.POLICY_GREEDY_BFS,
                                 input_policy.POLICY_GYM,
                             ))

    # for distributed DroidBot
    parser.add_argument("-distributed", action="store", dest="distributed", choices=["master", "worker"],
                        help="Start DroidBot in distributed mode.")
    parser.add_argument("-master", action="store", dest="master",
                        help="DroidMaster's RPC address")
    parser.add_argument("-qemu_hda", action="store", dest="qemu_hda",
                        help="The QEMU's hda image")
    parser.add_argument("-qemu_no_graphic", action="store_true", dest="qemu_no_graphic",
                        help="Run QEMU with -nograpihc parameter")

    parser.add_argument("-script", action="store", dest="script_path",
                        help="Use a script to customize input for certain states.")
    parser.add_argument("-count", action="store", dest="count", default=input_manager.DEFAULT_EVENT_COUNT, type=int,
                        help="Number of events to generate in total. Default: %d" % input_manager.DEFAULT_EVENT_COUNT)
    parser.add_argument("-interval", action="store", dest="interval", default=input_manager.DEFAULT_EVENT_INTERVAL,
                        type=int,
                        help="Interval in seconds between each two events. Default: %d" % input_manager.DEFAULT_EVENT_INTERVAL)
    parser.add_argument("-timeout", action="store", dest="timeout", default=input_manager.DEFAULT_TIMEOUT, type=int,
                        help="Timeout in seconds, -1 means unlimited. Default: %d" % input_manager.DEFAULT_TIMEOUT)
    parser.add_argument("-cv", action="store_true", dest="cv_mode",
                        help="Use OpenCV (instead of UIAutomator) to identify UI components. CV mode requires opencv-python installed.")
    parser.add_argument("-debug", action="store_true", dest="debug_mode",
                        help="Run in debug mode (dump debug messages).")
    parser.add_argument("-random", action="store_true", dest="random_input",
                        help="Add randomness to input events.")
    parser.add_argument("-keep_app", action="store_true", dest="keep_app",
                        help="Keep the app on the device after testing.")
    parser.add_argument("-keep_env", action="store_true", dest="keep_env",
                        help="Keep the test environment (eg. minicap and accessibility service) after testing.")
    parser.add_argument("-use_method_profiling", action="store", dest="profiling_method",
                        help="Record method trace for each event. can be \"full\" or a sampling rate.")
    parser.add_argument("-grant_perm", action="store_true", dest="grant_perm",
                        help="Grant all permissions while installing. Useful for Android 6.0+.")
    parser.add_argument("-is_emulator", action="store_true", dest="is_emulator",
                        help="Declare the target device to be an emulator, which would be treated specially by DroidBot.")
    parser.add_argument("-accessibility_auto", action="store_true", dest="enable_accessibility_hard",
                        help="Enable the accessibility service automatically even though it might require device restart\n(can be useful for Android API level < 23).")
    parser.add_argument("-humanoid", action="store", dest="humanoid",
                        help="Connect to a Humanoid service (addr:port) for more human-like behaviors.")
    parser.add_argument("-ignore_ad", action="store_true", dest="ignore_ad",
                        help="Ignore Ad views by checking resource_id.")
    parser.add_argument("-replay_output", action="store", dest="replay_output",
                        help="The droidbot output directory being replayed.")
    options = parser.parse_args()
    # print options
    return options


def main():
    """
    the main function
    it starts a droidbot according to the arguments given in cmd line
    """

    opts = parse_args()
    import os#操作系统库
    if not os.path.exists(opts.apk_path):
        print("APK does not exist.")
        return
    if not opts.output_dir and opts.cv_mode:
        print("To run in CV mode, you need to specify an output dir (using -o option).")

    if opts.distributed:
        if opts.distributed == "master":
            start_mode = "master"
        else:
            start_mode = "worker"
    else:
        start_mode = "normal"

    if start_mode == "master":
        droidmaster = DroidMaster(
            app_path=opts.apk_path,
            is_emulator=opts.is_emulator,
            output_dir=opts.output_dir,
            # env_policy=opts.env_policy,
            env_policy=env_manager.POLICY_NONE,
            policy_name=opts.input_policy,
            random_input=opts.random_input,
            script_path=opts.script_path,
            event_interval=opts.interval,
            timeout=opts.timeout,
            event_count=opts.count,
            cv_mode=opts.cv_mode,
            debug_mode=opts.debug_mode,
            keep_app=opts.keep_app,
            keep_env=opts.keep_env,
            profiling_method=opts.profiling_method,
            grant_perm=opts.grant_perm,
            enable_accessibility_hard=opts.enable_accessibility_hard,
            qemu_hda=opts.qemu_hda,
            qemu_no_graphic=opts.qemu_no_graphic,
            humanoid=opts.humanoid,
            ignore_ad=opts.ignore_ad,
            replay_output=opts.replay_output)
        droidmaster.start()
    else:
        droidbot = DroidBot(
            app_path=opts.apk_path,
            device_serial=opts.device_serial,
            is_emulator=opts.is_emulator,
            output_dir=opts.output_dir,
            # env_policy=opts.env_policy,
            env_policy=env_manager.POLICY_NONE,
            policy_name=opts.input_policy,
            random_input=opts.random_input,
            script_path=opts.script_path,
            event_interval=opts.interval,
            timeout=opts.timeout,
            event_count=opts.count,
            cv_mode=opts.cv_mode,
            debug_mode=opts.debug_mode,
            keep_app=opts.keep_app,
            keep_env=opts.keep_env,
            profiling_method=opts.profiling_method,
            grant_perm=opts.grant_perm,
            enable_accessibility_hard=opts.enable_accessibility_hard,
            master=opts.master,
            humanoid=opts.humanoid,
            ignore_ad=opts.ignore_ad,
            replay_output=opts.replay_output)

        droidbot.start()

    env = DummyVecEnv([lambda: droidbot_env.DroidBotEnv(droidbot)])
    start_time = time.time()
    env.reset()

    def events_so_state(env):
        events = env.envs[0].possible_events#操作
        print(len(events))
        print("events_so_state-events:",events)
        state_now = env.envs[0].device.get_current_state()#虚拟环境获取当前状态
        event_ids = []#存储事件id的列表容器
        probs = []#事件概率容器

        for i, event in enumerate(events):#遍历事件，确认当前事件是否出现过
            event_str = str(type(event)) + '_' + event.get_event_str(state_now)
            if event_str in event_ids:#如果event存在，1/0
                1/0#抛出一个 ZeroDivisionError 异常
            if event:
                event_ids.append(event_str)
                probs.append(env.envs[0].events_probs[i])
        state = state_now.state_str##获取当前android状态
        probs = np.array(probs)#把概率变矩阵
        print(event_ids)
        return state, probs, event_ids

    state_function = {}#状态，字典
    num_iterations = 1000
    EPSILON = 0.1#偏置
    Q_TABLE = []
    transitions_matrix = None
    number_of_trans = []
    event_to_id = []
    max_number_of_actions = 20#观察有没有差距
    max_number_of_states = 1000
    # 检查状态
    def check_state(state_id):
        # 检查状态是否存在于状态字典中，若不存在则初始化
        nonlocal Q_TABLE
        nonlocal transitions_matrix
        nonlocal number_of_trans
        nonlocal event_to_id
        nonlocal state_function
        #print(state_id)
        if state_function.get(state_id) is None:
            if Q_TABLE == []:#第一步
                Q_TABLE = np.zeros((1, max_number_of_actions))#0矩阵的初始化
                transitions_matrix = np.zeros((1, max_number_of_actions, 1))
            else:
                Q_TABLE = np.concatenate([Q_TABLE, np.zeros((1, max_number_of_actions))], axis=0)
                transition_matrix_new = np.zeros((Q_TABLE.shape[0], max_number_of_actions, Q_TABLE.shape[0]))
                transition_matrix_new[:-1, :, :-1] = transitions_matrix#把转移矩阵部分内容更新
                transitions_matrix = transition_matrix_new#更新转移矩阵
            event_to_id.append({})
            state_function[state_id] = Q_TABLE.shape[0] - 1#Q表行值-1
            Q_TABLE[-1][-1] = 1.0
            number_of_trans.append(np.zeros(max_number_of_actions))
        #print(state_function)
    #对state_pre 进行状态检查和初始化
    state_pre, probs, event_ids = events_so_state(env)
    check_state(state_pre)
    state = state_function[state_pre]

    from scipy.stats import beta

    # def plot(beta_parameters, trial):
    #     # 生成用于绘图的 x 值，创建一个包含 200 个在 0 到 1 之间均匀间隔的数值的数组
    #     x = np.linspace(0, 1, 200)
    #     # 遍历 Beta 分布参数列表中的每组参数
    #     for params in beta_parameters:
    #         # 从元组中提取形状参数 a 和 b
    #         a, b = params
    #         # 使用 Beta 分布参数计算每个 x 对应的概率密度函数（PDF）值
    #         y = beta.pdf(x, a, b)
    #         # 使用 plt.plot 绘制 Beta 分布曲线
    #         # 用真实概率的均值作为 Beta 分布的标签，真实概率计算为 Beta 分布的均值
    #         plt.plot(x, y, label="real P: %.4f" % (a / (a + b)))
    #     # 设置图的标题，指示试验次数
    #     plt.title("Bandit distributions after %s trials" % trial)
    #     # 为图添加图例以识别每条曲线
    #     plt.legend()
    #     # 显示图形
    #     plt.show()
    #     # 设置保存路径
    #     save_path = r'D:\Plot_picture'
    #     # 确保保存路径存在，如果不存在则创建
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     # 保存图形（你可以根据需要设置文件名和文件格式）
    #     save_file = os.path.join(save_path)
    #     plt.savefig(save_file)
    #     # 关闭图形窗口
    #     plt.close()


    event_distributions = {}
    state_event_matrix = np.zeros((max_number_of_states, max_number_of_actions))
    def make_decision(state_i, events, beta_parameters):
        nonlocal Q_TABLE, event_to_id, event_distributions
        # state_event_matrix = np.zeros((max_number_of_states, max_number_of_actions))

        id_to_action = np.zeros((max_number_of_actions), dtype=np.int32) + 1000
        q_values = np.zeros(max_number_of_actions)

        # 遍历 events，为每个 event 建立 Beta 分布
        beta_distributions = [beta(ba, bb) for ba, bb in beta_parameters]

        for i, event in enumerate(events):
            if i == len(events) - 1:
                q_values[-1] = Q_TABLE[state_i][-1]
                id_to_action[-1] = min(len(events), max_number_of_actions) - 1
                continue
            # print("event_to_id[state_i].get(event):",event_to_id[state_i].get(event))
            if event_to_id[state_i].get(event) is None:
                if len(event_to_id[state_i]) >= max_number_of_actions - 1:
                    continue
                event_to_id[state_i][event] = int(len(list(event_to_id[state_i].keys())))
                Q_TABLE[state_i][event_to_id[state_i][event]] = 1.0
            q_values[event_to_id[state_i][event]] = Q_TABLE[state_i][event_to_id[state_i][event]]
            id_to_action[event_to_id[state_i][event]] = int(i)

        # 为每个 event 采样 Beta 分布得到概率
        beta_probs = [beta_dist.rvs(size=1, random_state=None)[0] for beta_dist in beta_distributions]
        beta_probs /= np.sum(beta_probs)
        # 计算 1-最大采样概率
        epsilon = 1 - np.max(beta_probs)

        # 以概率选择动作
        action = np.argmax(beta_probs) if np.random.rand() > epsilon else np.random.choice(
            [i for i in range(max_number_of_actions) if i != np.argmax(beta_probs)])
        make_action = id_to_action[action]

        # 更新所有动作的 Beta 分布参数
        for i in range(max_number_of_actions):
            # 仅更新被选择动作以外的动作的 Beta 分布参数的第二个参数
            if i != action:
                beta_parameters[i] = (beta_parameters[i][0], max(1, beta_parameters[i][1] + 1))

        # 更新被选择动作的 Beta 分布参数的第一个参数
        beta_parameters[action] = (max(1, beta_parameters[action][0] + 1), beta_parameters[action][1])

        # 存储当前状态的事件分布
        event_distributions[state_i] = beta_probs.tolist()
        # 将当前状态的可能事件分布更新到矩阵中
        state_event_matrix[state_i] = beta_probs
        print("action:", action)
        print("make_action:" ,make_action)
        print("beta_parameters:", beta_parameters)
        print("event_distributions:", event_distributions)
        return action, make_action, beta_parameters

    # 初始化 Beta 分布参数和事件分布
    beta_parameters = [(1, 1) for _ in range(max_number_of_actions)]
    event_distributions = {}

    def update_q_values_with_event_distribution(Q_TABLE, transitions_matrix, state_event_matrix):
        for state_i in range(max_number_of_states):
            for event_j in range(max_number_of_actions):
                if state_event_matrix[state_i, event_j] > 0:
                    transitions = transitions_matrix[:, event_j, :]

                    q_target = np.array([[np.max(Q_TABLE[event_j])] for event_j in np.arange(Q_TABLE.shape[0])])
                    new_q_values = np.matmul(transitions, q_target) * 0.99
                    # 获取当前状态的事件分布概率
                    current_event_distribution = state_event_matrix[state_i]

                    # 计算事件分布均值
                    event_distribution_mean = np.mean(current_event_distribution)

                    # 加权平均更新 Q 值
                    new_q_values *= event_distribution_mean

                    # 归一化处理
                    new_q_values /= np.sum(new_q_values)
                    good_states = np.sum(transitions, axis=1) > 0.45
                    if True in good_states:
                        Q_TABLE[good_states, event_j] = new_q_values[good_states, 0]
    # 主循环
    for i_step in np.arange(num_iterations):
        action, make_action, beta_parameters = make_decision(state, event_ids, beta_parameters)
        print(i_step)
        env.step([make_action])

        new_state_pre, probs, event_ids = events_so_state(env)

        # 调用 plot 函数来观察每个赌博机的概率分布在试验次数增加时的变化。
        # plot(beta_parameters, i_step)
        check_state(new_state_pre)
        new_state = state_function[new_state_pre]
        # 更新 Q 表和状态转移矩阵
        number_of_trans[state][action] += 1
        transitions_matrix[state, action] *= (number_of_trans[state][action] - 1)
        transitions_matrix[state, action, new_state] += 1
        transitions_matrix[state, action] /= number_of_trans[state][action]

        # 更新 state_event_matrix
        state_event_matrix[state] = event_distributions[state]
        # 调用新函数更新 Q 值
        update_q_values_with_event_distribution(Q_TABLE, transitions_matrix, state_event_matrix)

        # for _ in np.arange(10):
        #     for i in np.arange(max_number_of_actions):
        #         transitions = transitions_matrix[:, i, :]
        #         q_target = np.array([[np.max(Q_TABLE[i])] for i in np.arange(Q_TABLE.shape[0])])
        #         new_q_values = np.matmul(transitions, q_target) * 0.99
        #         good_states = np.sum(transitions, axis=1) > 0.5
        #         if True in good_states:
        #             Q_TABLE[good_states, i] = new_q_values[good_states, 0]
        #         else:
        #             continue
        for i in np.arange(Q_TABLE.shape[0]):
            # print(Q_TABLE[i])
            rows = len(Q_TABLE)
            cols = len(Q_TABLE[0])
        print(rows,cols)
        # np.save('event_distributions.npy', event_distributions)
        if i_step%10==0:
            # 保存 Q 表、状态转移矩阵和状态字典
            np.save('q_function', Q_TABLE)
            np.save('transition_function', transitions_matrix)
            with open('states.json', 'w') as f:
                json.dump(state_function, f)
        state = new_state
    try:
        1 / 0
    except ZeroDivisionError:
        print("Error: Division by zero!")
    # 结束覆盖率测量
    cov.stop()

    # 保存覆盖率数据
    cov.save()

    # # 生成覆盖率报告
    # cov.report()

    # 生成 HTML 格式的覆盖率报告
    cov.html_report(directory='coverage_report')

    # 这个可以打印覆盖率详细信息，也可以注释掉
    cov.annotate()
    droidbot.stop()




if __name__ == "__main__":
    print("Starting droidbot gym env")
    main()
    # experiment()
