from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
from absl import logging
import tensorflow as tf

from tf_agents.environments import suite_mujoco, suite_gym
from tf_agents.environments import wrappers
from tf_agents.environments import batched_py_environment

from meld.environments import gym_wrappers
from meld.environments import video_wrapper
import IPython
e = IPython.embed

########################################
# load envs
########################################


def load_environments(universe, action_mode, env_name=None,
                      render_size=128, observation_render_size=64,
                      observations_whitelist=None, action_repeat=1,
                      num_train_tasks=30, num_eval_tasks=10, eval_on_holdout_tasks=True,
                      return_multiple_tasks=False, model_input=None,
                      auto_reset_task_each_episode=False,
                      ):

  """
  Loads train and eval environments.
  """

  assert universe == 'gym'
  tf.compat.v1.logging.info('Using environment {} from {} universe.'.format(env_name, universe))

  is_shelf_env = (env_name == 'SawyerShelfMT-v0') or (env_name == 'SawyerShelfMT-v2')
  if is_shelf_env:
    return load_multiple_mugs_env(universe, action_mode,
                                  env_name=env_name,
                                  observations_whitelist=['state', 'pixels',
                                                          'env_info'],
                                  action_repeat=action_repeat,
                                  num_train_tasks=num_train_tasks,
                                  num_eval_tasks=num_eval_tasks,
                                  eval_on_holdout_tasks=eval_on_holdout_tasks,
                                  return_multiple_tasks=True,
                                  )

  # select observation wrapper
  # puts either state or image into the 'pixels' location
  use_observation_wrapper = gym_wrappers.PixelObservationsGymWrapper
  if model_input is not None:
    if model_input=='state':
      use_observation_wrapper = gym_wrappers.PixelObservationsGymWrapperState
  
  # wrappers for train env (put on GPU 0)
  gym_env_wrappers = [
      functools.partial(gym_wrappers.RenderGymWrapper,
                        render_kwargs={'height': render_size,
                                       'width': render_size,
                                       'device_id': 0}),
      functools.partial(use_observation_wrapper,
                        observations_whitelist=observations_whitelist,
                        render_kwargs={'height': observation_render_size,
                                       'width': observation_render_size,
                                       'device_id': 0})]

  # wrappers for eval env (put on GPU 1)
  eval_gym_env_wrappers = [
      functools.partial(gym_wrappers.RenderGymWrapper,
                        render_kwargs={'height': render_size,
                                       'width': render_size,
                                       'device_id': 1}),
      # segfaults if the device is the same as train env
      functools.partial(use_observation_wrapper,
                        observations_whitelist=observations_whitelist,
                        render_kwargs={'height': observation_render_size,
                                       'width': observation_render_size,
                                       'device_id': 1})]  # segfaults if the device is the same as train env
  
  # create train/eval envs
  gym_kwargs = {"action_mode": action_mode}
  py_env = suite_gym.load(env_name, gym_env_wrappers=gym_env_wrappers, gym_kwargs=gym_kwargs)
  eval_py_env = suite_gym.load(env_name, gym_env_wrappers=eval_gym_env_wrappers, gym_kwargs=gym_kwargs)

  # set action mode
  py_env.wrapped_env().override_action_mode(action_mode)
  eval_py_env.wrapped_env().override_action_mode(action_mode)

  # video wrapper for eval saving
  eval_py_env = video_wrapper.VideoWrapper(eval_py_env)

  # action repeat
  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)
    eval_py_env = wrappers.ActionRepeat(eval_py_env, action_repeat)

  ###############################
  # get possible tasks
  ###############################

  if return_multiple_tasks:
    # set env as being "train" or "eval"
    # used for defining the tasks used in the envs
    eval_env_is_true_eval = False
    if eval_on_holdout_tasks:
      eval_env_is_true_eval = True

    # train env
    train_tasks = py_env.init_tasks(num_tasks=num_train_tasks, is_eval_env=False)
    # eval env
    eval_tasks = eval_py_env.init_tasks(num_tasks=num_eval_tasks, is_eval_env=eval_env_is_true_eval)


    # set task list and reset variable to true
    if auto_reset_task_each_episode:
      py_env.wrapped_env().set_auto_reset_task(train_tasks)
      eval_py_env.wrapped_env().set_auto_reset_task(eval_tasks)

    return py_env, eval_py_env, train_tasks, eval_tasks
  else:
    return py_env, eval_py_env


########################################
# helpers
########################################


def get_control_timestep(py_env):
  try:
    control_timestep = py_env.dt  # gym
  except AttributeError:
    control_timestep = py_env.control_timestep()  # dm_control
  return control_timestep


def get_wrappers(device_id, model_input, render_size, observation_render_size, observations_whitelist):
  use_observation_wrapper = gym_wrappers.PixelObservationsGymWrapper
  if model_input is not None:
    if model_input=='state':
      use_observation_wrapper = gym_wrappers.PixelObservationsGymWrapperState

  wrappers = [
    functools.partial(gym_wrappers.RenderGymWrapper,
                      render_kwargs={'height': render_size,
                                     'width': render_size,
                                     'device_id': device_id}),
    # segfaults if the device is the same as train env
    functools.partial(use_observation_wrapper,
                      observations_whitelist=observations_whitelist,
                      render_kwargs={'height': observation_render_size,
                                     'width': observation_render_size,
                                     'device_id': device_id})]  # segfaults if the device is the same as train env
  return wrappers

  
########################################
# specifically for mug env
########################################


def load_multiple_mugs_env(universe, action_mode, env_name=None,
                           render_size=128, observation_render_size=64,
                           observations_whitelist=None, action_repeat=1,
                           num_train_tasks=30, num_eval_tasks=10, eval_on_holdout_tasks=True,
                           return_multiple_tasks=False, model_input=None,
                           auto_reset_task_each_episode = False,
                           ):

  ### HARDCODED
  # temporary sanity
  assert env_name == 'SawyerShelfMT-v0'
  assert return_multiple_tasks
  assert universe == 'gym'

  # get eval and train tasks by loading a sample env
  sample_env = suite_mujoco.load(env_name)
  # train env
  train_tasks = sample_env.init_tasks(num_tasks=num_train_tasks, is_eval_env=False)
  # eval env
  eval_tasks = sample_env.init_tasks(num_tasks=num_eval_tasks, is_eval_env=eval_on_holdout_tasks)
  del sample_env

  print("train weights", train_tasks)
  print("eval weights", eval_tasks)
  if env_name == 'SawyerShelfMT-v0':
    from meld.environments.envs.shelf.assets.generate_sawyer_shelf_xml import generate_and_save_xml_file
  else:
    raise NotImplementedError

  train_xml_path = generate_and_save_xml_file(train_tasks, action_mode, is_eval=False)
  eval_xml_path = generate_and_save_xml_file(eval_tasks, action_mode, is_eval=True)

  ### train env
  # get wrappers
  wrappers = get_wrappers(device_id=0,
                          model_input=model_input,
                          render_size=render_size,
                          observation_render_size=observation_render_size,
                          observations_whitelist=observations_whitelist)
  # load env
  gym_kwargs = {"action_mode": action_mode, "xml_path": train_xml_path}
  py_env = suite_gym.load(env_name, gym_env_wrappers=wrappers, gym_kwargs=gym_kwargs)
  if action_repeat > 1:
    py_env = wrappers.ActionRepeat(py_env, action_repeat)

  ### eval env
  # get wrappers
  wrappers = get_wrappers(device_id=1,
                          model_input=model_input,
                          render_size=render_size,
                          observation_render_size=observation_render_size,
                          observations_whitelist=observations_whitelist)
  # load env
  gym_kwargs = {"action_mode": action_mode, "xml_path": eval_xml_path}
  eval_py_env = suite_gym.load(env_name, gym_env_wrappers=wrappers, gym_kwargs=gym_kwargs)
  eval_py_env = video_wrapper.VideoWrapper(eval_py_env)
  if action_repeat > 1:
    eval_py_env = wrappers.ActionRepeat(eval_py_env, action_repeat)

  py_env.assign_tasks(train_tasks)
  eval_py_env.assign_tasks(eval_tasks)

  # set task list and reset variable to true
  if auto_reset_task_each_episode:
    py_env.wrapped_env().set_auto_reset_task(train_tasks)
    eval_py_env.wrapped_env().set_auto_reset_task(eval_tasks)

  return py_env, eval_py_env, train_tasks, eval_tasks
