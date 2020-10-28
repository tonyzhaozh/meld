from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
from absl import app
from absl import flags
from absl import logging

# tf_agents
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.replay_buffers.episodic_replay_buffer import StatefulEpisodicReplayBuffer
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


# our custom "trial" driver 
# Note: a trial is multiple episodes of preserving the latent
# (instead of tf_agents.drivers.dynamic_episode_driver)
from meld.utils.env_drivers import DynamicTrialDriver

# networks
from meld.networks import actor_distribution_network
from meld.networks import critic_network
from meld.networks.meld_model_network import ModelDistributionNetwork

# MELD agent
from meld.agents.meld_agent import MeldAgent

# other utils
from meld.utils.logger import Logger
from meld.utils import custom_metrics
from meld.utils.utils import *
from meld.utils.env_utils import get_control_timestep, load_environments
from meld.utils.eval_summaries import perform_eval_and_summaries_meld
from meld.utils.eval_utils import load_eval_log

# env registration
from meld.environments.envs import register_all_gym_envs

# flags
flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name used for naming the output directory.')
flags.DEFINE_string('seed', '1', 'numpy random seed')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS

PRINT_TIMING = False


@gin.configurable
def train_eval(

    ##############################################
    # types of params:
    # 0: specific to algorithm (gin file 0)
    # 1: specific to environment (gin file 1)
    # 2: specific to experiment (gin file 2 + command line)
    
    # Note: there are other important params 
    # in eg ModelDistributionNetwork that the gin files specify
    # like sparse vs dense rewards, latent dimensions, etc.
    ##############################################

    # basic params for running/logging experiment
    root_dir,                                                                                         # 2
    experiment_name,                                                                                  # 2
    num_iterations=int(1e7),                                                                          # 2
    seed=1,                                                                                           # 2
    gpu_allow_growth=False,                                                                           # 2
    gpu_memory_limit=None,                                                                            # 2
    verbose=True,                                                                                     # 2
    policy_checkpoint_freq_in_iter=100, # policies needed for future eval                             # 2
    train_checkpoint_freq_in_iter=0, #default don't save                                              # 2
    rb_checkpoint_freq_in_iter=0, #default don't save                                                 # 2
    logging_freq_in_iter=10, # printing to terminal                                                   # 2
    summary_freq_in_iter=10, # saving to tb                                                           # 2
    num_images_per_summary=2,                                                                         # 2
    summaries_flush_secs=10,                                                                          # 2
    max_episode_len_override=None,                                                                    # 2
    num_trials_to_render=1,                                                                           # 2

    # environment, action mode, etc.
    env_name='HalfCheetah-v2',                                                                        # 1
    action_repeat=1,                                                                                  # 1
    action_mode='joint_position',  # joint_position or joint_delta_position                           # 1
    double_camera=False, # camera input                                                               # 1
    universe='gym',                                                                                   # default
    task_reward_dim=1,                                                                                # default

    # dims for all networks
    actor_fc_layers=(256, 256),                                                                       # 1
    critic_obs_fc_layers=None,                                                                        # 1
    critic_action_fc_layers=None,                                                                     # 1
    critic_joint_fc_layers=(256, 256),                                                                # 1
    num_repeat_when_concatenate=None,                                                                 # 1

    # networks
    critic_input='state',                                                                             # 0
    actor_input='state',                                                                              # 0

    # specifying tasks and eval
    episodes_per_trial = 1,                                                                           # 2
    num_train_tasks = 10,                                                                             # 2
    num_eval_tasks = 10,                                                                              # 2
    num_eval_trials = 10,                                                                             # 2
    eval_interval = 10,                                                                               # 2
    eval_on_holdout_tasks = True,                                                                     # 2

    # data collection/buffer
    init_collect_trials_per_task=None,                                                                # 2
    collect_trials_per_task=None,                                                                     # 2
    num_tasks_to_collect_per_iter=5,                                                                  # 2
    replay_buffer_capacity=int(1e5),                                                                  # 2

    # training
    init_model_train_ratio=0.8,                                                                       # 2
    model_train_ratio=1,                                                                              # 2
    model_train_freq=1,                                                                               # 2
    ac_train_ratio=1,                                                                                 # 2
    ac_train_freq=1,                                                                                  # 2
    num_tasks_per_train=5,                                                                            # 2
    train_trials_per_task=5,                                                                          # 2
    model_bs_in_steps=256,                                                                            # 2
    ac_bs_in_steps=128,                                                                               # 2

    # default AC learning rates, gamma, etc.
    target_update_tau=0.005,
    target_update_period=1,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    model_learning_rate=1e-4,
    td_errors_loss_fn=functools.partial(tf.compat.v1.losses.mean_squared_error, weights=0.5),
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    log_image_strips=False,
    stop_model_training=1E10,

    eval_only=False, # evaluate checkpoints ONLY
    log_image_observations=False,

    load_offline_data=False,  # whether to use offline data
    offline_data_dir=None,  # replay buffer's dir
    offline_episode_len=None, # episode len of episodes stored in rb
    offline_ratio=0, # ratio of data that is from offline buffer

  ):

  g = tf.Graph()

  # register all gym envs
  max_steps_dict = {"HalfCheetahVel-v0": 50, "SawyerReach-v0": 40, "SawyerReachMT-v0": 40,
                    "SawyerPeg-v0": 40, "SawyerPegMT-v0": 40, "SawyerPegMT4box-v0": 40,
                    "SawyerShelfMT-v0": 40, "SawyerKitchenMT-v0": 40, "SawyerShelfMT-v2": 40,
                    "SawyerButtons-v0": 40,
                    }
  if max_episode_len_override:
    max_steps_dict[env_name] = max_episode_len_override
  register_all_gym_envs(max_steps_dict)

  # set max_episode_len based on our env 
  max_episode_len = max_steps_dict[env_name]

  ######################################################
  # Calculate additional params
  ######################################################
  
  # convert to number of steps
  env_steps_per_trial = episodes_per_trial * max_episode_len
  real_env_steps_per_trial = episodes_per_trial * (max_episode_len + 1)
  env_steps_per_iter = num_tasks_to_collect_per_iter * collect_trials_per_task * env_steps_per_trial
  per_task_collect_steps = collect_trials_per_task*env_steps_per_trial

  # initial collect + train
  init_collect_env_steps = num_train_tasks * init_collect_trials_per_task * env_steps_per_trial
  init_model_train_steps = int(init_collect_env_steps * init_model_train_ratio)

  # collect + train
  collect_env_steps_per_iter = num_tasks_to_collect_per_iter * per_task_collect_steps
  model_train_steps_per_iter = int(env_steps_per_iter * model_train_ratio)
  ac_train_steps_per_iter = int(env_steps_per_iter * ac_train_ratio)
  
  # other
  global_steps_per_iter = collect_env_steps_per_iter + model_train_steps_per_iter + ac_train_steps_per_iter
  sample_episodes_per_task = train_trials_per_task * episodes_per_trial # number of episodes to sample from each replay
  model_bs_in_trials = model_bs_in_steps // real_env_steps_per_trial

  # assertions that make sure parameters make sense
  assert model_bs_in_trials > 0, "model batch size need to be at least as big as one full real trial"
  assert num_tasks_to_collect_per_iter <= num_train_tasks, "when sampling replace=False"
  assert num_tasks_per_train*train_trials_per_task >= model_bs_in_trials, "not enough data for one batch model train"
  assert num_tasks_per_train*train_trials_per_task*env_steps_per_trial >= ac_bs_in_steps, "not enough data for one batch ac train"

  ######################################################
  # Print a summary of params
  ######################################################
  MELD_summary_string = f"""\n\n\n
==============================================================
==============================================================
  \n
  MELD algorithm summary:

  * each trial consists of {episodes_per_trial} episodes
  * episode length: {max_episode_len}, trial length: {env_steps_per_trial}
  * {num_train_tasks} train tasks, {num_eval_tasks} eval tasks, hold-out: {eval_on_holdout_tasks}
  * environment: {env_name}
  
  For each of {num_train_tasks} tasks:
    Do {init_collect_trials_per_task} trials of initial collect
  (total {init_collect_env_steps} env steps)
  
  Do {init_model_train_steps} steps of initial model training
    
  For i in range(inf):
    For each of {num_tasks_to_collect_per_iter} randomly selected tasks:
      Do {collect_trials_per_task} trials of collect
    (which is {collect_trials_per_task*env_steps_per_trial} env steps per task)
    (for a total of {num_tasks_to_collect_per_iter*collect_trials_per_task*env_steps_per_trial} env steps in the iteration)
    
    if i % model_train_freq(={model_train_freq}):
      Do {model_train_steps_per_iter} steps of model training
        - select {sample_episodes_per_task} episodes from each of {num_tasks_per_train} random train_tasks, combine into {num_tasks_per_train*train_trials_per_task} total trials.
        - pick randomly {model_bs_in_trials} trials, train model on whole trials.
    
    if i % ac_train_freq(={ac_train_freq}):
      Do {ac_train_steps_per_iter} steps of ac training
        - select {sample_episodes_per_task} episodes from each of {num_tasks_per_train} random train_tasks, combine into {num_tasks_per_train*train_trials_per_task} total trials.
        - pick randomly {ac_bs_in_steps} transitions, not including between trial transitions, 
          to train ac.
  
  
  * Other important params:
  Evaluate policy every {eval_interval} iters, equivalent to {global_steps_per_iter*eval_interval/1000:.1f}k global steps
  Average evaluation across {num_eval_trials} trials
  Save summary to tensorboard every {summary_freq_in_iter} iters, equivalent to {global_steps_per_iter*summary_freq_in_iter/1000:.1f}k global steps
  Checkpoint:
   - training checkpoint every {train_checkpoint_freq_in_iter} iters, equivalent to {global_steps_per_iter*train_checkpoint_freq_in_iter//1000}k global steps, keep 1 checkpoint
   - policy checkpoint every {policy_checkpoint_freq_in_iter} iters, equivalent to {global_steps_per_iter*policy_checkpoint_freq_in_iter//1000}k global steps, keep all checkpoints
   - replay buffer checkpoint every {rb_checkpoint_freq_in_iter} iters, equivalent to {global_steps_per_iter*rb_checkpoint_freq_in_iter//1000}k global steps, keep 1 checkpoint
    
  \n
=============================================================
=============================================================
  """

  print(MELD_summary_string)
  time.sleep(1)

  ######################################################
  # Seed + name + GPU configs + directories for saving
  ######################################################
  np.random.seed(int(seed))
  experiment_name += "_seed" + str(seed)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpu_allow_growth:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpu_memory_limit:
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
          gpu,
          [tf.config.experimental.VirtualDeviceConfiguration(
              memory_limit=gpu_memory_limit)])

  train_eval_dir = get_train_eval_dir(
      root_dir, universe, env_name, experiment_name)
  train_dir = os.path.join(train_eval_dir, 'train')
  eval_dir = os.path.join(train_eval_dir, 'eval')
  eval_dir_2 = os.path.join(train_eval_dir, 'eval2')

  ######################################################
  # Train and Eval Summary Writers
  ######################################################
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_summary_flush_op = eval_summary_writer.flush()

  eval_logger = Logger(eval_dir_2)

  ######################################################
  # Train and Eval metrics
  ######################################################
  eval_buffer_size = num_eval_trials * episodes_per_trial * max_episode_len # across all eval trials in each evaluation
  eval_metrics = []
  for position in range(episodes_per_trial): # have metrics for each episode position, to track whether it is learning
    eval_metrics_pos = [
      py_metrics.AverageReturnMetric(
        name='c_AverageReturnEval_' + str(position),
        buffer_size=eval_buffer_size),
      py_metrics.AverageEpisodeLengthMetric(
        name='f_AverageEpisodeLengthEval_' + str(position),
        buffer_size=eval_buffer_size),
      custom_metrics.AverageScoreMetric(
        name="d_AverageScoreMetricEval_" + str(position),
        buffer_size=eval_buffer_size),
    ]
    eval_metrics.extend(eval_metrics_pos)

  train_buffer_size = num_train_tasks * episodes_per_trial
  train_metrics = [
    tf_metrics.NumberOfEpisodes(name='NumberOfEpisodes'),
    tf_metrics.EnvironmentSteps(name='EnvironmentSteps'),
    tf_py_metric.TFPyMetric(py_metrics.AverageReturnMetric(name="a_AverageReturnTrain", buffer_size=train_buffer_size)),
    tf_py_metric.TFPyMetric(py_metrics.AverageEpisodeLengthMetric(name="e_AverageEpisodeLengthTrain", buffer_size=train_buffer_size)),
    tf_py_metric.TFPyMetric(custom_metrics.AverageScoreMetric(name="b_AverageScoreTrain", buffer_size=train_buffer_size)),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step() # will be use to record number of model grad steps + ac grad steps + env_step

  log_cond = get_log_condition_tensor(global_step, init_collect_trials_per_task, env_steps_per_trial, num_train_tasks,
                                      init_model_train_steps, collect_trials_per_task, num_tasks_to_collect_per_iter,
                                      model_train_steps_per_iter, ac_train_steps_per_iter, summary_freq_in_iter,
                                      eval_interval)

  with tf.compat.v2.summary.record_if(log_cond):

    ######################################################
    # Create env
    ######################################################
    py_env, eval_py_env, train_tasks, eval_tasks = load_environments(universe, action_mode,
                                                                     env_name=env_name,
                                                                     observations_whitelist=['state', 'pixels',
                                                                                             "env_info"],
                                                                     action_repeat=action_repeat,
                                                                     num_train_tasks=num_train_tasks,
                                                                     num_eval_tasks=num_eval_tasks,
                                                                     eval_on_holdout_tasks=eval_on_holdout_tasks,
                                                                     return_multiple_tasks=True,
                                                                     )
    override_reward_func = None
    if load_offline_data:
      py_env.set_task_dict(train_tasks)
      override_reward_func = py_env.override_reward_func

    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=True)

    # Get data specs from env
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()
    original_control_timestep = get_control_timestep(eval_py_env)

    # fps
    control_timestep = original_control_timestep * float(action_repeat)
    render_fps = int(np.round(1.0 / original_control_timestep))

    ######################################################
    # Latent variable model
    ######################################################
    if verbose:
      print("-- start constructing model networks --")

    model_net = ModelDistributionNetwork(double_camera=double_camera,
                                         observation_spec=observation_spec,
                                         num_repeat_when_concatenate=num_repeat_when_concatenate,
                                         task_reward_dim=task_reward_dim,
                                         episodes_per_trial=episodes_per_trial,
                                         max_episode_len=max_episode_len) # rest of arguments provided via gin

    if verbose:
      print("-- finish constructing AC networks --")

    ######################################################
    # Compressor Network for Actor/Critic
    # The model's compressor is also used by the AC
    # compressor function: images --> features
    ######################################################

    compressor_net = model_net.compressor

    ######################################################
    # Specs for Actor and Critic
    ######################################################
    if actor_input == 'state':
      actor_state_size = observation_spec['state'].shape[0]
    elif actor_input == 'latentSample':
      actor_state_size = model_net.state_size
    elif actor_input == "latentDistribution":
      actor_state_size = 2 * model_net.state_size # mean and (diagonal) variance of gaussian, of two latents
    else:
      raise NotImplementedError
    actor_input_spec = tensor_spec.TensorSpec((actor_state_size, ), dtype=tf.float32)

    if critic_input == 'state':
      critic_state_size = observation_spec['state'].shape[0]
    elif critic_input == 'latentSample':
      critic_state_size = model_net.state_size
    elif critic_input == "latentDistribution":
      critic_state_size = 2 * model_net.state_size # mean and (diagonal) variance of gaussian, of two latents
    else:
      raise NotImplementedError
    critic_input_spec = tensor_spec.TensorSpec((critic_state_size, ), dtype=tf.float32)

    ######################################################
    # Actor and Critic Networks
    ######################################################
    if verbose:
      print("-- start constructing Actor and Critic networks --")

    actor_net = actor_distribution_network.ActorDistributionNetwork(
      actor_input_spec,
      action_spec,
      fc_layer_params=actor_fc_layers,)

    critic_net = critic_network.CriticNetwork(
      (critic_input_spec, action_spec),
      observation_fc_layer_params=critic_obs_fc_layers,
      action_fc_layer_params=critic_action_fc_layers,
      joint_fc_layer_params=critic_joint_fc_layers)

    if verbose:
      print("-- finish constructing AC networks --")
      print("-- start constructing agent --")

    ######################################################
    # Create the agent
    ######################################################

    which_posterior_overwrite = None
    which_reward_overwrite = None

    meld_agent = MeldAgent(
      # specs
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      # step counter
      train_step_counter=global_step, # will count number of model training steps
      # networks
      actor_network=actor_net,
      critic_network=critic_net,
      model_network=model_net,
      compressor_network=compressor_net,
      # optimizers
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
      model_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=model_learning_rate),
      # target update
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      # inputs
      critic_input=critic_input,
      actor_input=actor_input,
      # bs stuff
      model_batch_size=model_bs_in_steps,
      ac_batch_size = ac_bs_in_steps,
      # other
      num_tasks_per_train=num_tasks_per_train,
      td_errors_loss_fn=td_errors_loss_fn,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      control_timestep=control_timestep,
      num_images_per_summary=num_images_per_summary,
      task_reward_dim=task_reward_dim,
      episodes_per_trial=episodes_per_trial,
      # offline data
      override_reward_func=override_reward_func,
      offline_ratio=offline_ratio,
    )

    if verbose:
      print("-- finish constructing agent --")

    ######################################################
    # Replay buffers + observers to add data to them
    ######################################################
    replay_buffers = []
    replay_observers = []
    for _ in range(num_train_tasks):
      replay_buffer_episodic = episodic_replay_buffer.EpisodicReplayBuffer(
        meld_agent.collect_policy.trajectory_spec, # spec of each point stored in here (i.e. Trajectory)
        capacity=replay_buffer_capacity,
        completed_only=True, # in as_dataset, if num_steps is None, this means return full episodes
        # device='GPU:0', # gpu not supported for some reason
        begin_episode_fn=lambda traj: traj.is_first()[0], # first step of seq we add should be is_first
        end_episode_fn=lambda traj: traj.is_last()[0], # last step of seq we add should be is_last
        dataset_drop_remainder=True,  #`as_dataset` makes the final batch be dropped if it does not contain exactly `sample_batch_size` items
      )
      replay_buffer = StatefulEpisodicReplayBuffer(replay_buffer_episodic) # adding num_episodes here is bad
      replay_buffers.append(replay_buffer)
      replay_observers.append([replay_buffer.add_sequence])

    if load_offline_data:
      # for each task, has a separate replay buffer for relabeled data
      replay_buffers_withRelabel = []
      replay_observers_withRelabel = []
      for _ in range(num_train_tasks):
        replay_buffer_episodic_withRelabel = episodic_replay_buffer.EpisodicReplayBuffer(
          meld_agent.collect_policy.trajectory_spec,  # spec of each point stored in here (i.e. Trajectory)
          capacity=replay_buffer_capacity,
          completed_only=True,  # in as_dataset, if num_steps is None, this means return full episodes
          # device='GPU:0', # gpu not supported for some reason
          begin_episode_fn=lambda traj: traj.is_first()[0],  # first step of seq we add should be is_first
          end_episode_fn=lambda traj: traj.is_last()[0],  # last step of seq we add should be is_last
          dataset_drop_remainder=True,
          # `as_dataset` makes the final batch be dropped if it does not contain exactly `sample_batch_size` items
        )
        replay_buffer_withRelabel = StatefulEpisodicReplayBuffer(replay_buffer_episodic_withRelabel)  # adding num_episodes here is bad
        replay_buffers_withRelabel.append(replay_buffer_withRelabel)
        replay_observers_withRelabel.append([replay_buffer_withRelabel.add_sequence])

    if verbose:
      print("-- finish constructing replay buffers --")
      print("-- start constructing policies and collect ops --")

    ######################################################
    # Policies
    #####################################################

    # init collect policy (random)
    init_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)

    # eval
    eval_py_policy = py_tf_policy.PyTFPolicy(meld_agent.policy)

    ################################################################################
    # Collect ops : use policies to get data + have the observer put data into corresponding RB
    ################################################################################

    #init collection (with random policy)
    init_collect_ops = []
    for task_idx in range(num_train_tasks):
      # put init data into the rb + track with the train metric
      observers = replay_observers[task_idx] + train_metrics

      # initial collect op
      init_collect_op = DynamicTrialDriver(
        tf_env,
        init_collect_policy,
        num_trials_to_collect=init_collect_trials_per_task,
        observers=observers,
        episodes_per_trial=episodes_per_trial,  # policy state will not be reset within these episodes
        max_episode_len=max_episode_len,
      ).run() # collect one trial
      init_collect_ops.append(init_collect_op)

    # data collection for training (with collect policy)
    collect_ops = []
    for task_idx in range(num_train_tasks):
      collect_op = DynamicTrialDriver(
        tf_env,
        meld_agent.collect_policy,
        num_trials_to_collect=collect_trials_per_task,
        observers=replay_observers[task_idx] + train_metrics,  # put data into 1st RB + track with 1st pol metrics
        episodes_per_trial=episodes_per_trial,  # policy state will not be reset within these episodes
        max_episode_len=max_episode_len,
      ).run() # collect one trial
      collect_ops.append(collect_op)

    if verbose:
      print("-- finish constructing policies and collect ops --")
      print("-- start constructing replay buffer->training pipeline --")

    ######################################################
    # replay buffer --> dataset --> iterate to get trajecs for training
    ######################################################

    # get some data from all task replay buffers (even though won't actually train on all of them)
    dataset_iterators = []
    all_tasks_trajectories_fromdense = []
    for task_idx in range(num_train_tasks):
      dataset = replay_buffers[task_idx].as_dataset(
        sample_batch_size=sample_episodes_per_task, # number of episodes to sample
        num_steps=max_episode_len+1).prefetch(3) # +1 to include the last state: a trajectory with n transition has n+1 states
      # iterator to go through the data
      dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
      dataset_iterators.append(dataset_iterator)
      # get sample_episodes_per_task sequences, each of length num_steps
      trajectories_task_i, _ = dataset_iterator.get_next()
      all_tasks_trajectories_fromdense.append(trajectories_task_i)

    if load_offline_data:
      # have separate dataset for relabel data
      dataset_iterators_withRelabel = []
      all_tasks_trajectories_fromdense_withRelabel = []
      for task_idx in range(num_train_tasks):
        dataset = replay_buffers_withRelabel[task_idx].as_dataset(
          sample_batch_size=sample_episodes_per_task, # number of episodes to sample
          num_steps=offline_episode_len+1).prefetch(3) # +1 to include the last state: a trajectory with n transition has n+1 states
        # iterator to go through the data
        dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        dataset_iterators_withRelabel.append(dataset_iterator)
        # get sample_episodes_per_task sequences, each of length num_steps
        trajectories_task_i, _ = dataset_iterator.get_next()
        all_tasks_trajectories_fromdense_withRelabel.append(trajectories_task_i)






    if verbose:
      print("-- finish constructing replay buffer->training pipeline --")
      print("-- start constructing model and AC training ops --")

    ######################################
    # Decoding latent samples into rewards
    ######################################

    latent_samples_1_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, meld_agent._model_network.latent1_size)) 
    latent_samples_2_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, meld_agent._model_network.latent2_size))
    decode_rews_op = meld_agent._model_network.decode_latents_into_reward(latent_samples_1_ph, latent_samples_2_ph)

    ######################################
    # Model/Actor/Critic train + summary ops
    ######################################
    

    # train AC on data from replay buffer
    if load_offline_data:
      ac_train_op = meld_agent.train_ac_meld(all_tasks_trajectories_fromdense, all_tasks_trajectories_fromdense_withRelabel)
    else:
      ac_train_op = meld_agent.train_ac_meld(all_tasks_trajectories_fromdense)

    summary_ops = []
    for train_metric in train_metrics:
      summary_ops.append(train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2]))

    if verbose:
      print("-- finish constructing AC training ops --")

    ############################
    # Model train + summary ops
    ############################

    # train model on data from replay buffer
    if load_offline_data:
      model_train_op, check_step_types = meld_agent.train_model_meld(all_tasks_trajectories_fromdense, all_tasks_trajectories_fromdense_withRelabel)
    else:
      model_train_op, check_step_types = meld_agent.train_model_meld(all_tasks_trajectories_fromdense)

    model_summary_ops, model_summary_ops_2 = [], []
    for summary_op in tf.compat.v1.summary.all_v2_summary_ops():
      if summary_op not in summary_ops:
        model_summary_ops.append(summary_op)

    if verbose:
      print("-- finish constructing model training ops --")
      print("-- start constructing checkpointers --")

    ########################
    # Eval metrics
    ########################

    with eval_summary_writer.as_default(), \
         tf.compat.v2.summary.record_if(True):
      for eval_metric in eval_metrics:
        eval_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])

    ########################
    # Create savers
    ########################
    train_config_saver = gin.tf.GinConfigSaverHook(train_dir, summarize_config=False)
    eval_config_saver = gin.tf.GinConfigSaverHook(eval_dir, summarize_config=False)

    ########################
    # Create checkpointers
    ########################

    train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=meld_agent,
      global_step=global_step,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
      max_to_keep=1)
    policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      policy=meld_agent.policy,
      global_step=global_step,
      max_to_keep=99999999999) # keep many policy checkpoints, in case of future eval
    rb_checkpointers = []
    for buffer_idx in range(len(replay_buffers)):
      rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffers/', "task" + str(buffer_idx)),
        max_to_keep=1,
        replay_buffer=replay_buffers[buffer_idx])
      rb_checkpointers.append(rb_checkpointer)

    if load_offline_data: # for LOADING data not for checkpointing. No new data going in anyways
      rb_checkpointers_withRelabel = []
      for buffer_idx in range(len(replay_buffers_withRelabel)):
        ckpt_dir = os.path.join(offline_data_dir, "task" + str(buffer_idx))
        rb_checkpointer = common.Checkpointer(
          ckpt_dir=ckpt_dir,
          max_to_keep=99999999999,
          replay_buffer=replay_buffers_withRelabel[buffer_idx])
        rb_checkpointers_withRelabel.append(rb_checkpointer)
      # Notice: these replay buffers need to follow the same sequence of tasks as the current one

    if verbose:
      print("-- finish constructing checkpointers --")
      print("-- start main training loop --")

    with tf.compat.v1.Session() as sess:

      ########################
      # Initialize
      ########################

      if eval_only:
        sess.run(eval_summary_writer.init())
        load_eval_log(train_eval_dir=train_eval_dir,
                  meld_agent=meld_agent,
                  global_step=global_step,
                  sess=sess,
                  eval_metrics=eval_metrics,
                  eval_py_env=eval_py_env,
                  eval_py_policy=eval_py_policy,
                  num_eval_trials=num_eval_trials,
                  max_episode_len=max_episode_len,
                  episodes_per_trial=episodes_per_trial,
                  log_image_strips=log_image_strips,
                  num_trials_to_render=num_trials_to_render,
                  train_tasks=train_tasks, # in case want to eval on a train task
                  eval_tasks=eval_tasks,
                  model_net=model_net,
                  render_fps=render_fps,
                  decode_rews_op=decode_rews_op,
                  latent_samples_1_ph=latent_samples_1_ph,
                  latent_samples_2_ph=latent_samples_2_ph,
                  )
        return

      # Initialize checkpointing
      train_checkpointer.initialize_or_restore(sess)
      for rb_checkpointer in rb_checkpointers:
        rb_checkpointer.initialize_or_restore(sess)

      if load_offline_data:
        for rb_checkpointer in rb_checkpointers_withRelabel:
          rb_checkpointer.initialize_or_restore(sess)

      # Initialize dataset iterators
      for dataset_iterator in dataset_iterators:
        sess.run(dataset_iterator.initializer)

      if load_offline_data:
        for dataset_iterator in dataset_iterators_withRelabel:
          sess.run(dataset_iterator.initializer)


      # Initialize variables
      common.initialize_uninitialized_variables(sess)

      # Initialize summary writers
      sess.run(train_summary_writer.init())
      sess.run(eval_summary_writer.init())

      # Initialize savers
      train_config_saver.after_create_session(sess)
      eval_config_saver.after_create_session(sess)
      # Get value of step counter
      global_step_val = sess.run(global_step)

      if verbose:
        print("====== finished initialization ======")

      ################################################################
      # If this is start of new exp (i.e., 1st step) and not continuing old exp
      # eval rand policy + do initial data collection
      ################################################################
      fresh_start = (global_step_val == 0)

      if fresh_start:

        ########################
        # Evaluate initial policy
        ########################

        if eval_interval:
          logging.info('\n\nDoing evaluation of initial policy on %d trials with randomly sampled tasks', num_eval_trials)
          perform_eval_and_summaries_meld(
            eval_metrics,
            eval_py_env,
            eval_py_policy,
            num_eval_trials,
            max_episode_len,
            episodes_per_trial,
            log_image_strips=log_image_strips,
            num_trials_to_render=num_eval_tasks,
            eval_tasks=eval_tasks,
            latent1_size=model_net.latent1_size,
            latent2_size=model_net.latent2_size,
            logger=eval_logger,
            global_step_val=global_step_val,
            render_fps=render_fps,
            decode_rews_op=decode_rews_op,
            latent_samples_1_ph=latent_samples_1_ph,
            latent_samples_2_ph=latent_samples_2_ph,
            log_image_observations=log_image_observations,
            )
          sess.run(eval_summary_flush_op)
          logging.info('Done with evaluation of initial (random) policy.\n\n')

        ########################
        # Initial data collection
        ########################

        logging.info('\n\nGlobal step %d: Beginning init collect op with random policy. Collecting %dx {%d, %d} trials for each task',
                     global_step_val, init_collect_trials_per_task, max_episode_len, episodes_per_trial)

        init_increment_global_step_op = global_step.assign_add(env_steps_per_trial * init_collect_trials_per_task)

        for task_idx in range(num_train_tasks):
          logging.info('on task %d / %d', task_idx+1, num_train_tasks)
          py_env.set_task_for_env(train_tasks[task_idx])
          sess.run([init_collect_ops[task_idx], init_increment_global_step_op]) # incremented gs in granularity of task

        rb_checkpointer.save(global_step=global_step_val)
        logging.info('Finished init collect.\n\n')

      else:
        logging.info('\n\nGlobal step %d from loaded experiment: Skipping init collect op.\n\n', global_step_val)

      #########################
      # Create calls
      #########################

      # [1] calls for running the policies to collect training data
      collect_calls = []
      increment_global_step_op = global_step.assign_add(env_steps_per_trial * collect_trials_per_task)
      for task_idx in range(num_train_tasks):
        collect_calls.append(sess.make_callable([collect_ops[task_idx], increment_global_step_op]))

      # [2] call for doing a training step (A + C)
      ac_train_step_call = sess.make_callable([ac_train_op, summary_ops])

      # [3] call for doing a training step (model)
      model_train_step_call = sess.make_callable([model_train_op, check_step_types, model_summary_ops])

      # [4] call for evaluating what global_step number we're on
      global_step_call = sess.make_callable(global_step)

      # reset keeping track of steps/time
      timed_at_step = global_step_call()
      time_acc = 0
      steps_per_second_ph = tf.compat.v1.placeholder(tf.float32, shape=(), name='steps_per_sec_ph')
      with train_summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
        steps_per_second_summary = tf.compat.v2.summary.scalar(
          name='global_steps_per_sec', data=steps_per_second_ph,
          step=global_step)

      #################################
      # init model training
      #################################
      if fresh_start:
        logging.info(
          '\n\nPerforming %d steps of init model training, each step on %d random tasks', init_model_train_steps, num_tasks_per_train)
        for i in range(init_model_train_steps):

          temp_start = time.time()
          if i % 100 == 0:
            print(".... init model training ", i, "/", init_model_train_steps)

          # init model training
          total_loss_value_model, check_step_types, _ = model_train_step_call()

          if PRINT_TIMING:
            print("single model train step: ", time.time() - temp_start)

      if verbose:
        print("\n\n\n-- start training loop --\n")

      #################################
      # Training Loop
      #################################
      start_time = time.time()
      for iteration in range(num_iterations):

        if iteration>0:
          g.finalize()

        # print("\n\n\niter", iteration, sess.run(curr_iter))
        print("global step", global_step_call())

        logging.info("Iteration: %d, Global step: %d\n", iteration, global_step_val)

        ####################
        # collect data
        ####################
        logging.info('\nStarting batch data collection. Collecting %d {%d, %d} trials for each of %d tasks',
                     collect_trials_per_task, max_episode_len, episodes_per_trial, num_tasks_to_collect_per_iter)

        # randomly select tasks to collect this iteration
        list_of_collect_task_idxs = np.random.choice(len(train_tasks), num_tasks_to_collect_per_iter, replace=False)
        for count, task_idx in enumerate(list_of_collect_task_idxs):
          logging.info('on randomly selected task %d / %d', count+1, num_tasks_to_collect_per_iter)

          # set task for the env
          py_env.set_task_for_env(train_tasks[task_idx])

          # collect data with collect policy
          _, policy_state_val = collect_calls[task_idx]()

        logging.info('Finish data collection. Global step: %d\n', global_step_call())

        ####################
        # train model
        ####################
        if (iteration == 0) or ((iteration % model_train_freq == 0) and (global_step_val < stop_model_training)):
          logging.info('\n\nPerforming %d steps of model training, each on %d random tasks', model_train_steps_per_iter, num_tasks_per_train)
          for model_iter in range(model_train_steps_per_iter):
            temp_start_2 = time.time()

            # train model
            total_loss_value_model, _, _ = model_train_step_call()

            # print("is logging step", model_iter, sess.run(is_logging_step))
            if PRINT_TIMING:
              print("2: single model train step: ", time.time() - temp_start_2)
          logging.info('Finish model training. Global step: %d\n', global_step_call())
        else:
          print("SKIPPING MODEL TRAINING")

        ####################
        # train actor critic
        ####################
        if iteration % ac_train_freq == 0:
          logging.info('\n\nPerforming %d steps of AC training, each on %d random tasks \n\n', ac_train_steps_per_iter, num_tasks_per_train)
          for ac_iter in range(ac_train_steps_per_iter):
            temp_start_2_ac = time.time()

            # train ac
            total_loss_value_ac, _ = ac_train_step_call()
            if PRINT_TIMING:
              print("2: single AC train step: ", time.time() - temp_start_2_ac)
        logging.info('Finish AC training. Global step: %d\n', global_step_call())

        # add up time
        time_acc += time.time() - start_time

        ####################
        # logging/summaries
        ####################

        ### Eval
        if eval_interval and (iteration % eval_interval == 0):
          logging.info('\n\nDoing evaluation of trained policy on %d trials with randomly sampled tasks', num_eval_trials)

          perform_eval_and_summaries_meld(
            eval_metrics,
            eval_py_env,
            eval_py_policy,
            num_eval_trials,
            max_episode_len,
            episodes_per_trial,
            log_image_strips=log_image_strips,
            num_trials_to_render=num_trials_to_render, # hardcoded: or gif will get too long
            eval_tasks=eval_tasks,
            latent1_size=model_net.latent1_size,
            latent2_size=model_net.latent2_size,
            logger=eval_logger,
            global_step_val=global_step_call(),
            render_fps=render_fps,
            decode_rews_op=decode_rews_op,
            latent_samples_1_ph=latent_samples_1_ph,
            latent_samples_2_ph=latent_samples_2_ph,
            log_image_observations=log_image_observations,
            )

        ### steps_per_second_summary
        global_step_val = global_step_call()
        if logging_freq_in_iter and (iteration % logging_freq_in_iter == 0):
          # log step number + speed (steps/sec)
          logging.info('step = %d, loss = %f', global_step_val, total_loss_value_ac.loss + total_loss_value_model.loss)
          steps_per_sec = (global_step_val - timed_at_step) / time_acc
          logging.info('%.3f env_steps/sec', steps_per_sec)
          sess.run(steps_per_second_summary, feed_dict={steps_per_second_ph: steps_per_sec})

          # reset keeping track of steps/time
          timed_at_step = global_step_val
          time_acc = 0

        ### train_checkpoint
        if train_checkpoint_freq_in_iter and (iteration % train_checkpoint_freq_in_iter == 0):
          train_checkpointer.save(global_step=global_step_val)

        ### policy_checkpointer
        if policy_checkpoint_freq_in_iter and (iteration % policy_checkpoint_freq_in_iter == 0):
          policy_checkpointer.save(global_step=global_step_val)

        ### rb_checkpointer
        if rb_checkpoint_freq_in_iter and (iteration % rb_checkpoint_freq_in_iter == 0):
          for rb_checkpointer in rb_checkpointers:
            rb_checkpointer.save(global_step=global_step_val)


#################################
# Main
#################################

def main(argv):

  # set logging/etc.
  tf.compat.v1.enable_resource_variables()
  FLAGS(argv)  # raises UnrecognizedFlagError for undefined flags
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  # read variables from configs
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=False)
  # run
  train_eval(FLAGS.root_dir, FLAGS.experiment_name, seed=FLAGS.seed)

if __name__ == '__main__':
  app.run(main)