from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re
import time

import copy
import gin
import gin.tf
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from tf_agents.trajectories import trajectory
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.trajectories import time_step as ts

from meld.utils import gif_utils
from tf_agents.utils import eager_utils


def get_log_condition_tensor(global_step, init_collect_trials_per_task, env_steps_per_trial, num_train_tasks,
                             init_model_train_steps, collect_trials_per_task, num_tasks_to_collect_per_iter,
                             model_train_steps_per_iter, ac_train_steps_per_iter, summary_freq_in_iter, eval_interval):

  """
  want to record summary to tensorboard when:
  - At the end of initial training op     -> log_cond_initial
  - At the end of initial model training  -> log_cond_initial
  - Every summary_freq_in_iter in training loop, record model, ac training and collection stats  -> log_cond_normal
  - Every time compute_summary is called  -> log_cond_eval
  """

  init_per_task_collect_steps = init_collect_trials_per_task*env_steps_per_trial
  init_collect_env_steps = num_train_tasks*init_per_task_collect_steps
  init_total_steps = init_collect_env_steps + init_model_train_steps
  initial_phase = tf.math.less_equal(global_step, init_total_steps)
  init_collect_last_task = tf.math.equal(global_step, (num_train_tasks-1)*init_per_task_collect_steps)
  init_train_last_step = tf.math.equal(global_step, init_total_steps-1)

  per_task_collect_steps = collect_trials_per_task*env_steps_per_trial
  collect_env_steps = num_tasks_to_collect_per_iter*per_task_collect_steps
  steps_per_iter = collect_env_steps + model_train_steps_per_iter + ac_train_steps_per_iter

  curr_iter = tf.math.maximum((global_step - init_total_steps) // steps_per_iter, 0)
  is_logging_iter = tf.math.equal(tf.math.floormod(curr_iter, summary_freq_in_iter), 0)
  step_within_iter = global_step - init_total_steps - (curr_iter * steps_per_iter)
  collect_last_task = tf.math.equal(step_within_iter, (num_tasks_to_collect_per_iter-1)*per_task_collect_steps)
  train_model_last_step = tf.math.equal(step_within_iter-collect_env_steps, model_train_steps_per_iter-1)
  train_ac_last_step = tf.math.equal(step_within_iter-collect_env_steps-model_train_steps_per_iter, ac_train_steps_per_iter-1)
  is_logging_step = tf.math.logical_or(collect_last_task, tf.math.logical_or(train_model_last_step, train_ac_last_step))

  is_first_eval = tf.math.equal(global_step, 0)
  is_eval_iter = tf.math.equal(tf.math.floormod(curr_iter-1, eval_interval), 0)
  is_eval_step = tf.math.equal(step_within_iter, 0)
  is_normal_eval = tf.math.logical_and(is_eval_iter, is_eval_step)

  log_cond_initial = tf.math.logical_and(initial_phase, tf.math.logical_or(init_collect_last_task, init_train_last_step))
  log_cond_normal = tf.math.logical_and(is_logging_iter, is_logging_step)
  log_cond_eval = tf.math.logical_or(is_first_eval, is_normal_eval)

  log_cond = tf.math.logical_or(log_cond_initial, tf.math.logical_or(log_cond_normal, log_cond_eval))

  return log_cond






########################################
# directory for saving
########################################


def get_train_eval_dir(root_dir, universe, env_name, experiment_name):

  root_dir = os.path.expanduser(root_dir)
  if universe == 'gym':
    train_eval_dir = os.path.join(root_dir, universe, env_name,
                                  experiment_name)
  else:
    raise ValueError('Invalid universe %s.' % universe)
  return train_eval_dir


################################################
# Convert experience (trajec) to transitions
################################################


def experience_to_transitions(experience):
  # convert
  transitions = trajectory.to_transition(experience)
  time_steps, policy_steps, next_time_steps = transitions
  actions = policy_steps.action
  # (s, a, s')
  return time_steps, actions, next_time_steps


################################
# Generic function to apply gradients
################################


def apply_gradients(gradients, variables, optimizer, gradient_clipping):
  grads_and_vars = zip(gradients, variables)
  if gradient_clipping is not None:
    grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars, gradient_clipping)
  optimizer.apply_gradients(grads_and_vars)


########################################
# gif and video summaries 
########################################


def _gif_summary(name, images, fps, saturate=False, step=None):
  images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
  output = tf.concat(tf.unstack(images), axis=2)[None]
  gif_utils.gif_summary_v2(name, output, 1, fps, step=step)


def _gif_and_image_summary(name, images, fps, saturate=False, step=None):
  images = tf.image.convert_image_dtype(images, tf.uint8, saturate=saturate)
  output = tf.concat(tf.unstack(images), axis=2)[None]
  gif_utils.gif_summary_v2(name, output, 1, fps, step=step)
  output = tf.concat(tf.unstack(images), axis=2)
  output = tf.concat(tf.unstack(output), axis=0)[None]
  tf.contrib.summary.image(name, output, step=step)


def pad_and_concatenate_videos(videos):
  videos.reverse() # left -> right: early -> late episodes
  max_episode_length = max([len(video) for video in videos])
  for video in videos:
    if len(video) < max_episode_length:
      video.extend(
          [np.zeros_like(video[-1])] * (max_episode_length - len(video)))
  videos = [np.concatenate(frames, axis=1) for frames in zip(*videos)]
  return videos


########################################
# filtering
########################################


def filter_before_first_step(time_steps, actions=None):
  flat_time_steps = tf.nest.flatten(time_steps)
  flat_time_steps = [tf.unstack(time_step, axis=1) for time_step in
                     flat_time_steps]
  time_steps = [tf.nest.pack_sequence_as(time_steps, time_step) for time_step in
                zip(*flat_time_steps)]
  if actions is None:
    actions = [None] * len(time_steps)
  else:
    actions = tf.unstack(actions, axis=1)
  assert len(time_steps) == len(actions)

  time_steps = list(reversed(time_steps))
  actions = list(reversed(actions))
  filtered_time_steps = []
  filtered_actions = []
  for t, (time_step, action) in enumerate(zip(time_steps, actions)):
    if t == 0:
      reset_mask = tf.equal(time_step.step_type, ts.StepType.FIRST)
    else:
      time_step = tf.nest.map_structure(lambda x, y: tf.where(reset_mask, x, y),
                                        last_time_step, time_step)
      action = tf.where(reset_mask, tf.zeros_like(action),
                        action) if action is not None else None
    filtered_time_steps.append(time_step)
    filtered_actions.append(action)
    reset_mask = tf.logical_or(
        reset_mask,
        tf.equal(time_step.step_type, ts.StepType.FIRST))
    last_time_step = time_step
  filtered_time_steps = list(reversed(filtered_time_steps))
  filtered_actions = list(reversed(filtered_actions))

  filtered_flat_time_steps = [tf.nest.flatten(time_step) for time_step in
                              filtered_time_steps]
  filtered_flat_time_steps = [tf.stack(time_step, axis=1) for time_step in
                              zip(*filtered_flat_time_steps)]
  filtered_time_steps = tf.nest.pack_sequence_as(filtered_time_steps[0],
                                                 filtered_flat_time_steps)
  if action is None:
    return filtered_time_steps
  else:
    actions = tf.stack(filtered_actions, axis=1)
    return filtered_time_steps, actions


################################################
# random helpers
################################################


def mask_episode_transition(time_step):
  masked_time_step = TimeStep(step_type=StepType.MID, reward=time_step.reward, discount=time_step.discount,
                              observation=time_step.observation)
  return masked_time_step


def flatten(input, axis=1, end_axis=-1):
  """
  Caffe-style flatten.

  Args:
    inputs: An N-D tensor.
    axis: The first axis to flatten: all preceding axes are retained in the
      output. May be negative to index from the end (e.g., -1 for the last
      axis).
    end_axis: The last axis to flatten: all following axes are retained in the
      output. May be negative to index from the end (e.g., the default -1 for
      the last axis)
  Returns:
      A M-D tensor where M = N - (end_axis - axis)
  """
  input_shape = tf.shape(input)
  input_rank = tf.shape(input_shape)[0]
  if axis < 0:
    axis = input_rank + axis
  if end_axis < 0:
    end_axis = input_rank + end_axis
  output_shape = []
  if axis != 0:
    output_shape.append(input_shape[:axis])
  output_shape.append([tf.reduce_prod(input_shape[axis:end_axis + 1])])
  if end_axis + 1 != input_rank:
    output_shape.append(input_shape[end_axis + 1:])
  output_shape = tf.concat(output_shape, axis=0)
  output = tf.reshape(input, output_shape)
  return output


def tf_random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")