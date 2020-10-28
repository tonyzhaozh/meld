import numpy as np
from gym.envs.mujoco import mujoco_env

class HalfCheetahVelEnv(mujoco_env.MujocoEnv):

    def __init__(self, n_tasks=3, randomize_tasks=True, action_mode=None):
        self.debug = False
        self.target_vel = 0  # placeholder
        self.truncate_vel_diff = 0.5
        self.frame_skip = 10
        self.step_counter = 0
        self.auto_reset_task = False
        self.auto_reset_task_list = None

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', self.frame_skip)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        self.step_counter += 1
        action*=0.8

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.target_vel)  ### The reward is related to the target velocity
        ctrl_cost = 0.01 * np.sum(np.square(action))
        vel_error = abs(forward_vel - self.target_vel)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, task=self.target_vel)

        if vel_error < self.truncate_vel_diff:
            sparse_reward = reward
        else:
            sparse_reward = 0

        observation = np.concatenate((observation, np.array([self.step_counter]), np.array([sparse_reward]), np.array([reward])))

        score = -vel_error
        infos = np.array([score, infos["reward_ctrl"], forward_vel, self.target_vel, 0])

        return (observation, reward, done, infos)

    def reset(self):
        # original mujoco
        ob = self.reset_model()
        self.step_counter = 0
        ob = np.concatenate((ob, np.array([self.step_counter]), np.array([0]), np.array([0])))

        # RESET task every episode, randomly
        if self.auto_reset_task:
            task_idx = np.random.randint(len(self.auto_reset_task_list))
            self.set_task_for_env(self.auto_reset_task_list[task_idx])
            
        return ob

    def init_tasks(self, num_tasks, is_eval_env):
        """To be called externally to obtain samples from the task distribution"""
        # set seed
        if is_eval_env:
            np.random.seed(100) #pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        min_vel = 0
        max_vel = 3
        velocities = np.random.uniform(min_vel, max_vel, size=(num_tasks,))

        print(f"Sampled goal velocities:", velocities)
        return velocities

    def set_task_for_env(self, goal):
        """To be called externally to set the task for this environment"""
        self.target_vel = goal

    def override_action_mode(self, action_mode):
        pass

    def set_auto_reset_task(self, task_list):
        self.auto_reset_task = True 
        self.auto_reset_task_list = task_list