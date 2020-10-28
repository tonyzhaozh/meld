from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from meld.policies.meld_policy import MeldPolicy
from meld.utils.utils import _gif_summary, tf_random_choice, apply_gradients
from meld.utils.train_utils import alpha_loss_fn, critic_loss_fn, actor_loss_fn, model_loss_fn

@gin.configurable
class MeldAgent(tf_agent.TFAgent):

  def __init__(self,

               # counter
               train_step_counter,

               # specs
               time_step_spec,
               action_spec,

               # networks
               critic_network,
               actor_network,
               model_network,
               compressor_network,

               # optimizers
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               model_optimizer,

               # target update
               target_update_tau=1.0,
               target_update_period=1,

               # inputs and stop gradients
               critic_input='state',
               actor_input='state',
               critic_input_stop_gradient=True,
               actor_input_stop_gradient=False,

               # model stuff
               model_batch_size=256, # will round to nearest full trajectory
               ac_batch_size=128,

               # other
               episodes_per_trial = 1,
               num_tasks_per_train=1,
               num_batches_per_sampled_trials=1,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               task_reward_dim=None,
               initial_log_alpha=0.0,
               target_entropy=None,
               gradient_clipping=None,
               control_timestep=None,
               num_images_per_summary=1,

               offline_ratio=None,
               override_reward_func=None,
               ):

    tf.Module.__init__(self)
    self.override_reward_func = override_reward_func
    self.offline_ratio = offline_ratio

    ################
    # critic
    ################
    # networks
    self._critic_network1 = critic_network
    self._critic_network2 = critic_network.copy(name='CriticNetwork2')
    self._target_critic_network1 = critic_network.copy(name='TargetCriticNetwork1')
    self._target_critic_network2 = critic_network.copy(name='TargetCriticNetwork2')
    # update the target networks
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._update_target = self._get_target_updater(tau=self._target_update_tau, period=self._target_update_period)

    ################
    # model
    ################
    self._model_network = model_network
    self.model_input = self._model_network.model_input

    ################
    # compressor
    ################
    self._compressor_network = compressor_network

    ################
    # actor
    ################
    self._actor_network = actor_network

    ################
    # policies
    ################

    self.condition_on_full_latent_dist = (actor_input=="latentDistribution" and critic_input=="latentDistribution")
    
    # both policies below share the same actor network
    # but they process latents (to give to actor network) in potentially different ways

    # used for eval
    which_posterior='first'
    if self._model_network.sparse_reward_inputs:
      which_rew_input='sparse'
    else:
      which_rew_input='dense'

    policy = MeldPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        model_network=self._model_network,
        actor_input=actor_input,
        which_posterior=which_posterior,
        which_rew_input=which_rew_input,
        )

    # used for collecting data during training

    # overwrite if specified (eg for double agent)
    which_posterior='first'
    if self._model_network.sparse_reward_inputs:
      which_rew_input='sparse'
    else:
      which_rew_input='dense'

    collect_policy = MeldPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=self._actor_network,
      model_network=self._model_network,
      actor_input=actor_input,
      which_posterior=which_posterior,
      which_rew_input=which_rew_input,
      )


    ################
    # more vars
    ################
    self.num_batches_per_sampled_trials = num_batches_per_sampled_trials
    self.episodes_per_trial = episodes_per_trial
    self._task_reward_dim = task_reward_dim
    self._log_alpha = common.create_variable(
        'initial_log_alpha',
        initial_value=initial_log_alpha,
        dtype=tf.float32,
        trainable=True)

    # If target_entropy was not passed, set it to negative of the total number
    # of action dimensions.
    if target_entropy is None:
      flat_action_spec = tf.nest.flatten(action_spec)
      target_entropy = -np.sum([
        np.product(single_spec.shape.as_list())
        for single_spec in flat_action_spec
      ])

    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self._alpha_optimizer = alpha_optimizer
    self._model_optimizer = model_optimizer
    self._td_errors_loss_fn = td_errors_loss_fn
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_entropy = target_entropy
    self._gradient_clipping = gradient_clipping

    self._critic_input = critic_input
    self._actor_input = actor_input
    self._critic_input_stop_gradient = critic_input_stop_gradient
    self._actor_input_stop_gradient = actor_input_stop_gradient
    self._model_batch_size = model_batch_size
    self._ac_batch_size = ac_batch_size
    self._control_timestep = control_timestep
    self._num_images_per_summary = num_images_per_summary
    self._actor_time_step_spec = time_step_spec._replace(observation=actor_network.input_tensor_spec)
    self._num_tasks_per_train = num_tasks_per_train

    ################
    # init tf agent
    ################

    super(MeldAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=collect_policy, #used to set self.step_spec
        train_sequence_length=None, #train function can accept experience of any length T (i.e., [B,T,...])
        train_step_counter=train_step_counter)

    self._train_model_fn = common.function_in_tf1()(self._train_model)
    self._train_ac_fn = common.function_in_tf1()(self._train_ac)

  ################################################
  # Copy weights from the Q networks to the target Q networks
  ################################################

  def _initialize(self):
    common.soft_variables_update(
        self._critic_network1.variables,
        self._target_critic_network1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._critic_network2.variables,
        self._target_critic_network2.variables,
        tau=1.0)

  #############################################
  # Soft update of target network params
  # w_t = (1- tau) x w_t + tau x ws
  #############################################
  
  def _get_target_updater(self, tau=1.0, period=1):
    scope = 'update_target'
    with tf.name_scope(scope):
      def update():
        """Update target network."""
        critic_update_1 = common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables, tau)
        critic_update_2 = common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables, tau)
        return tf.group(critic_update_1, critic_update_2)
      return common.Periodically(update, period, 'update_targets')


  #############################################
  # Utility function
  #############################################

  def concat_episodes(self, experiences_as_tensor, experiences_as_tensor_additional=None):

    has_additional_data = (experiences_as_tensor_additional is not None)

    # pick num_tasks_per_train random tasks to train on
    num_incoming_tasks = experiences_as_tensor.step_type.shape[0].value
    if has_additional_data:
      assert self._num_tasks_per_train == num_incoming_tasks, "train on all tasks for now. Make it easier to implement relabeling"
      experiences_for_training = experiences_as_tensor # select ALL tasks follow train_tasks' sequence of tasks
    else:
      rand_task_indices = tf.random.uniform(shape=[self._num_tasks_per_train], maxval=num_incoming_tasks, dtype=tf.int32)
      experiences_for_training = tf.nest.map_structure(lambda x: tf.gather(x, rand_task_indices), experiences_as_tensor)

    # Decompose the Trajectory object into individual tensors
    # each has shape (num_tasks_per_train, batch_size_per_task, real_episode_len, ...)
    step_types = experiences_for_training.step_type
    observation_states = experiences_for_training.observation["state"]
    observation_pixels = experiences_for_training.observation["pixels"]
    observation_env_infos = experiences_for_training.observation["env_info"]
    actions = experiences_for_training.action
    next_step_types = experiences_for_training.next_step_type
    rewards = experiences_for_training.reward
    discounts = experiences_for_training.discount
    # policy_info is ignored since it is ()
    all_data = [step_types, observation_states, observation_pixels, observation_env_infos, actions,
                next_step_types, rewards, discounts]

    if has_additional_data:
      experiences_for_training_additional = experiences_as_tensor_additional
      # Decompose the Trajectory object into individual tensors
      # each has shape (num_tasks_per_train, batch_size_per_task, real_episode_len, ...)
      step_types = experiences_for_training_additional.step_type
      observation_states = experiences_for_training_additional.observation["state"]
      observation_pixels = experiences_for_training_additional.observation["pixels"]
      observation_env_infos = experiences_for_training_additional.observation["env_info"]
      actions = experiences_for_training_additional.action
      next_step_types = experiences_for_training_additional.next_step_type
      rewards = experiences_for_training_additional.reward
      discounts = experiences_for_training_additional.discount
      all_data_additional = [step_types, observation_states, observation_pixels, observation_env_infos, actions,
                  next_step_types, rewards, discounts]
      # policy_info is ignored since it is ()

      # fix the episode len if offline and online data has different len
      online_episode_len = all_data[0].shape[-1].value
      offline_episode_len = all_data_additional[0].shape[-1].value
      assert offline_episode_len >= online_episode_len
      if online_episode_len != offline_episode_len:
        # truncate from the beginning
        all_data_additional = list(map(lambda d: d[:, :, :online_episode_len], all_data_additional))

      # notice ONLY additional data needs to be relabeled. The on-policy data will be using the same sparse reward setting
      all_data_additional = self.override_reward_func(all_data_additional) # the returned tensors will be of same shape

      # Combine the normal data with additional data
      # - Generate a mask over episode axis for EACH task
      # - construct the inverse mask
      # - new data = normal_data * inv_mask + additional_data * mask
      # this effectively injects additional data in the middle of normal data
      # when we do tf.split to obtain trials: additional data can appear anywhere
      num_train_tasks = experiences_for_training.step_type.shape[0].value
      episodes_per_task = experiences_for_training.step_type.shape[1].value
      base_mask_shape = [num_train_tasks, episodes_per_task]
      rand_num = tf.random.uniform(shape=base_mask_shape, maxval=1, dtype=tf.dtypes.float16)
      base_mask = rand_num < self.offline_ratio # if True: then use additional data

      max_dim = max([len(d.shape.as_list()) for d in all_data])
      bm = base_mask
      base_masks = [bm]
      for i in range(max_dim-2):
        bm = tf.expand_dims(bm, axis=-1)
        base_masks.append(bm)

      combined_data = []
      for normal_data, additional_data in zip(all_data, all_data_additional):
        data_shape = normal_data.shape.as_list()
        additional_dim = len(data_shape) - 2
        mask = tf.broadcast_to(base_masks[additional_dim], data_shape)
        mask = tf.cast(mask, normal_data.dtype)
        inv_mask = 1 - mask

        comb_data = mask * additional_data + inv_mask * normal_data
        combined_data.append(comb_data)

      all_data = combined_data

    # Note: all elements in all_data has shape (num_tasks_per_train, batch_size_per_task, real_episode_len, ...)

    num_tasks_per_train, batch_size_per_task, real_episode_len = all_data[0].shape.as_list()
    # Note:
    # num_tasks_per_train = train_eval.num_tasks_per_train
    #     - how many tasks does the data come from
    # batch_size_per_task = train_eval.train_trials_per_task * episodes_per_trial
    #     - how many episodes do we get from each task's replay buffer
    # real_episode_len = train_eval.max_episode_len + 1
    #     - we need the boundary transitions for MELD, which are normally thrown away

    # Concatenate episodes from same task to form trials
    episodes_per_trial = self.episodes_per_trial
    assert batch_size_per_task % episodes_per_trial == 0, "sampled #episodes from each task replay shall form full trials"
    num_trials_per_task = batch_size_per_task // episodes_per_trial
    # Concatenate each chunk of episodes_per_trial episodes to a single long episode
    for i, d in enumerate(all_data):
      data_shape = d.shape.as_list()
      individual_shape = data_shape[3:]  # shape of individual element
      raw_trials = tf.split(d, num_trials_per_task, axis=1) #(num_tasks_per_train*num_trials, episodes_per_trial*real_episode_len, ...)
      concat_trials = [
        tf.reshape(raw_trial, [num_tasks_per_train, episodes_per_trial * real_episode_len] + individual_shape) for
        raw_trial in raw_trials]
      all_data[i] = tf.concat(concat_trials, axis=0) #(num_tasks_per_train*num_trials_per_task, episodes_per_trial*real_episode_len, ...)

    total_num_trials = num_trials_per_task * num_tasks_per_train
    real_env_steps_per_trial = episodes_per_trial*real_episode_len
    all_spec = (total_num_trials, real_env_steps_per_trial, real_episode_len)

    return all_data, all_spec
    # step_types, observation_states, observation_pixels, observation_env_infos, actions, next_step_types, rewards, discounts = all_data


  #############################################
  # Train model, given batched experiences
  #############################################

  def train_model_meld(self, all_experiences, all_experiences_additional=[], weights=None):
    # check for existence of train model function
    if self._enable_functions and getattr(self, "_train_model_fn", None) is None:
      raise RuntimeError("Cannot find _train_model_fn.  Did you call super?")

    # call training
    if self._enable_functions:
      loss_info, check_step_types = self._train_model_fn(all_experiences, all_experiences_additional, weights=weights)
    else:
      loss_info, check_step_types = self._train_model(all_experiences, all_experiences_additional, weights=weights)

    # check type of loss + return it
    if not isinstance(loss_info, tf_agent.LossInfo):
      raise TypeError("loss_info is not a subclass of LossInfo: {}".format(loss_info))
    return loss_info, check_step_types

  def _train_model(self, all_experiences, all_experiences_additional, weights=None):

    print("\n\n====== Starting to set up model train pipeline ======")
    
    for _ in range(self.num_batches_per_sampled_trials):

      with tf.GradientTape() as tape:

        if all_experiences_additional: # has offline data
          # NOTICE: naive formation of trials: no real and offline/relabel data will be put into same trial
          # turn in the incoming list into a tensor
          experiences_as_tensor = tf.nest.pack_sequence_as(all_experiences[0], [tf.stack(items) for items in zip(*[tf.nest.flatten(trajectories) for trajectories in all_experiences])])
          experiences_as_tensor_additional = tf.nest.pack_sequence_as(all_experiences_additional[0], [tf.stack(items) for items in zip(*[tf.nest.flatten(trajectories) for trajectories in all_experiences_additional])])

          all_data, all_spec = self.concat_episodes(experiences_as_tensor, experiences_as_tensor_additional) # concat episodes to form trials
          total_num_trials, real_env_steps_per_trial, episode_len = all_spec

        else:
          # turn in the incoming list into a tensor
          experiences_as_tensor = tf.nest.pack_sequence_as(all_experiences[0], [tf.stack(items) for items in zip(
            *[tf.nest.flatten(trajectories) for trajectories in all_experiences])])
          all_data, all_spec = self.concat_episodes(experiences_as_tensor)  # concat episodes to form trials
          total_num_trials, real_env_steps_per_trial, episode_len = all_spec


        bs_num_trials = self._model_batch_size // real_env_steps_per_trial

        # pick only some of the trials to train model
        all_trial_indices = tf.constant(list(range(total_num_trials)), dtype=tf.int32)
        selected_trial_indices = tf_random_choice(all_trial_indices, bs_num_trials)
        data_use = tf.nest.map_structure(lambda x: tf.gather(x, selected_trial_indices), all_data)

        # prepare argument for calling model_loss
        step_types, observation_states, observation_pixels, observation_env_infos, actions, next_step_types, rewards, discounts = data_use

        # set model input, do not provide it non-necessary information
        if self.model_input == 'image':
          images = tf.image.convert_image_dtype(observation_pixels, tf.float32)
          augmented_states = observation_states  # in MELD augmented_state is state concat with sparse and dense rewards
        elif self.model_input == 'state':
          images = None
          augmented_states = observation_states  # in MELD augmented_state is state concat with sparse and dense rewards
        else:
          raise NotImplementedError

        model_loss, outputs = model_loss_fn(images, augmented_states, actions, step_types, self.model_input,
                  self._model_network, weights=weights) # model network will decide to use sparse or dense

        model_loss = (model_loss/self._num_tasks_per_train)


      #####################################
      # summaries of scalars and videos
      #####################################

      # note these summaries are being done on
      # only the last task seen during this train call
      for name, output in outputs.items():
        if output.shape.ndims == 0:
          tf.contrib.summary.scalar(name, output)
        elif output.shape.ndims == 5:
          fps = 10 if self._control_timestep is None else int(np.round(1.0 / self._control_timestep))
          _gif_summary(name, output[:self._num_images_per_summary], fps, saturate=True, step=self.train_step_counter)
        else:
          raise NotImplementedError

      ###############################
      # Model variables + gradients
      ###############################
      tf.debugging.check_numerics(model_loss, 'Model loss is inf or nan.')
      model_variables = list(self._model_network.variables)
      assert model_variables, 'No model variables to optimize.'
      model_grads = tape.gradient(model_loss, model_variables)

      for grad_idx in range(len(model_grads)):
        tf.debugging.check_numerics(model_grads[grad_idx], 'Model grads are inf or nan but the loss itself was fine')
      apply_gradients(model_grads, model_variables, self._model_optimizer, self._gradient_clipping)

      ###############################
      # Summarize loss + increase counter
      ###############################
      scope = 'Losses'
      with tf.name_scope(scope):
        tf.compat.v2.summary.scalar(name='model_loss', data=model_loss, step=self.train_step_counter)
      self.train_step_counter.assign_add(1)

    return tf_agent.LossInfo(loss=model_loss, extra=()), []


  ################################################################################################
  ################################################################################################
  # Train actor + critic, given batched experiences
  ################################################################################################
  ################################################################################################

  def train_ac_meld(self, all_experiences, all_experiences_additional=[], weights=None):
    # check for existence of train model function
    if self._enable_functions and getattr(self, "_train_ac_fn", None) is None:
      raise RuntimeError("Cannot find _train_ac_fn.  Did you call super?")

    # call training
    if self._enable_functions:
      loss_info = self._train_ac_fn(all_experiences, all_experiences_additional, weights=weights)
    else:
      loss_info = self._train_ac(all_experiences, all_experiences_additional, weights=weights)

    # check type of loss + return it
    if not isinstance(loss_info, tf_agent.LossInfo):
      raise TypeError("loss_info is not a subclass of LossInfo: {}".format(loss_info))
    return loss_info

  def _train_ac(self, all_experiences, all_experiences_additional, weights=None):
    print("\n\n====== Starting to set up AC train pipeline ======")

    # in order to train the AC:
    # the actor/critic network needs
    # - observations: in MELD it means latentDist or latentSample
    # - actions, step_types, ...

    for _ in range(self.num_batches_per_sampled_trials):

      with tf.GradientTape(persistent=True) as tape:

        if all_experiences_additional: # has offline data
          # NOTICE: naive formation of trials: no real and offline/relabel data will be put into same trial
          # turn in the incoming list into a tensor
          experiences_as_tensor = tf.nest.pack_sequence_as(all_experiences[0], [tf.stack(items) for items in zip(*[tf.nest.flatten(trajectories) for trajectories in all_experiences])])
          experiences_as_tensor_additional = tf.nest.pack_sequence_as(all_experiences_additional[0], [tf.stack(items) for items in zip(*[tf.nest.flatten(trajectories) for trajectories in all_experiences_additional])])

          all_data, all_spec = self.concat_episodes(experiences_as_tensor, experiences_as_tensor_additional) # concat episodes to form trials
          total_num_trials, real_env_steps_per_trial, episode_len = all_spec

        else:
          # turn in the incoming list into a tensor
          experiences_as_tensor = tf.nest.pack_sequence_as(all_experiences[0], [tf.stack(items) for items in zip(
            *[tf.nest.flatten(trajectories) for trajectories in all_experiences])])
          all_data, all_spec = self.concat_episodes(experiences_as_tensor)  # concat episodes to form trials
          total_num_trials, real_env_steps_per_trial, episode_len = all_spec

          # images = tf.image.convert_image_dtype(observation_pixels, tf.float32)  # (15, 25, 64, 64, 3)
        step_types, observation_states, observation_pixels, observation_env_infos, actions, next_step_types, rewards, discounts = all_data

        images = tf.image.convert_image_dtype(observation_pixels, tf.float32)  # (15, 25, 64, 64, 3)

        #####################################
        # Sample LVM to get latents
        #####################################

        print("    Proprocessing inputs to set up [im-->latents]")

        augmented_state = observation_states

        latent_samples, latent_dists = self._model_network.sample_prior_or_posterior(actions,
                                                                              step_types=step_types,
                                                                              images=images,
                                                                              augmented_states=augmented_state) 

        if isinstance(latent_samples, (tuple, list)):
          latent_samples = tf.concat(latent_samples, axis=-1)
        # process latent dist to [mu_1, sigma_1, mu_2, sigma_2] array
        latent1_dists = latent_dists[0]
        latent2_dists = latent_dists[1]
        processed_latent_dists_list = [latent1_dists.mean(), latent1_dists.stddev(), latent2_dists.mean(), latent2_dists.stddev()]
        processed_latent_dists = tf.concat(processed_latent_dists_list, axis=-1)

        all_data = [step_types, observation_states, actions, next_step_types, rewards, discounts, latent_samples, processed_latent_dists]

        # Note:
        # - processed_latent_dist: (total_num_trials, real_env_steps_per_trial, (z1_shape+z2_shape)*2))
        # - all_data: each has (total_num_trials, real_env_steps_per_trial, individual_shape)

        # collapse the first two dimensions: (total_num_trials, real_env_steps_per_trial)
        num_transitions = total_num_trials * real_env_steps_per_trial

        for i, d in enumerate(all_data):
          orig_shape = d.shape.as_list()
          individual_shape = orig_shape[2:] # per rollout
          all_data[i] = tf.reshape(d, [num_transitions] + individual_shape)

        [step_types, observation_states, actions, next_step_types, rewards, discounts, latent_samples, processed_latent_dists] = all_data

        del rewards # will not be used, instead will use reward from observation_states
        augmented_states = observation_states
        _, dense_rewards = self._model_network.extract_sparse_dense_reward(augmented_states)
        dense_rewards = tf.squeeze(dense_rewards)

        # Select randomly a subset of data to train on
        # all_trainable_indices = [i for i in range(num_transitions-1) if ((i+1) % real_env_steps_per_trial != 0)] # avoid "trial boundary" transitions
        all_trainable_indices = [i for i in range(num_transitions-1) if (((i+1) % real_env_steps_per_trial != 0) and ((i+1) % episode_len !=0 ))] # avoid "trial boundary" AND "episode boundary" transitions
        all_trainable_indices_tensor = tf.constant(all_trainable_indices, dtype=tf.int32)

        transition_bs = self._ac_batch_size
        selected_indices = tf_random_choice(all_trainable_indices_tensor, transition_bs)
        offset_selected_indices = selected_indices + 1

        curr_states, curr_latent_samples, curr_latent_dists, actions, dense_rewards = tf.nest.map_structure(lambda x: tf.gather(x, selected_indices), [observation_states, latent_samples, processed_latent_dists, actions, dense_rewards])
        next_states, next_latent_samples, next_latent_dists, next_discounts = tf.nest.map_structure(lambda x: tf.gather(x, offset_selected_indices), [observation_states, latent_samples, processed_latent_dists, discounts])
        step_types, next_step_types = tf.nest.map_structure(lambda x: tf.gather(x, selected_indices), [step_types, next_step_types])


        # Assign correct variable to "observation" of actor and critic
        # latent sample
        if self._critic_input == self._actor_input == 'latentSample':
          observations = curr_latent_samples
          next_observations = next_latent_samples
        # latent distribution
        elif self._critic_input == self._actor_input == "latentDistribution":
          observations = curr_latent_dists
          next_observations = next_latent_dists
        else:
          raise NotImplementedError

        # Note:
        # each argument passed into loss_fn has shape (transition_bs, individual_dim)

        #############################################
        # Critic/Actor/Alpha losses
        #############################################

        # Note: critic always gets dense rewards
        # (because it's only used during train time)
        critic_loss = critic_loss_fn(observations, next_observations, step_types, next_step_types, dense_rewards,
                                          next_discounts, actions,
                                          td_errors_loss_fn=self._td_errors_loss_fn,
                                          critic_input_stop_gradient=self._critic_input_stop_gradient,
                                          log_alpha=self._log_alpha, actor_network=self._actor_network,
                                          target_critic_network1=self._target_critic_network1,
                                          target_critic_network2=self._target_critic_network2,
                                          critic_network1=self._critic_network1, critic_network2=self._critic_network2,
                                          gamma=self._gamma, reward_scale_factor=self._reward_scale_factor,
                                          weights=weights)
        critic_loss = (critic_loss / self._num_tasks_per_train)

        actor_loss = actor_loss_fn(observations, step_types,
                                        actor_network=self._actor_network, critic_network1=self._critic_network1, critic_network2=self._critic_network2,
                                        log_alpha=self._log_alpha, actor_input_stop_gradient=self._actor_input_stop_gradient, weights=weights)
        actor_loss = (actor_loss/self._num_tasks_per_train)

        alpha_loss_i = alpha_loss_fn(observations, step_types,
                                          actor_network=self._actor_network,
                                          log_alpha=self._log_alpha, target_entropy=self._target_entropy, weights=weights)
        alpha_loss = (alpha_loss_i/self._num_tasks_per_train)

      #############################################
      # Critic variables + gradients
      #############################################
      tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
      critic_variables = (
          list(self._critic_network1.variables) +
          list(self._critic_network2.variables) +
          list(self._compressor_network.variables) +
          list(self._model_network.variables))
      assert critic_variables, 'No critic variables to optimize.'
      critic_grads = tape.gradient(critic_loss, critic_variables)
      apply_gradients(critic_grads, critic_variables, self._critic_optimizer, self._gradient_clipping)

      #############################################
      # Actor/alpha variables + gradients
      #############################################
      tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
      actor_variables = (
          list(self._actor_network.variables) +
          list(self._compressor_network.variables) +
          list(self._model_network.variables))
      assert actor_variables, 'No actor variables to optimize.'
      actor_grads = tape.gradient(actor_loss, actor_variables)
      apply_gradients(actor_grads, actor_variables, self._actor_optimizer, self._gradient_clipping)

      tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
      alpha_variables = [self._log_alpha]
      assert alpha_variables, 'No alpha variable to optimize.'
      alpha_grads = tape.gradient(alpha_loss, alpha_variables)
      apply_gradients(alpha_grads, alpha_variables, self._alpha_optimizer, self._gradient_clipping)

      #############################################
      # All losses
      #############################################
      total_loss = critic_loss + actor_loss + alpha_loss

      scope = 'Losses'
      with tf.name_scope(scope):
        tf.compat.v2.summary.scalar(name='critic_loss', data=critic_loss, step=self.train_step_counter)
        tf.compat.v2.summary.scalar(name='actor_loss', data=actor_loss, step=self.train_step_counter)
        tf.compat.v2.summary.scalar(name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

      #############################################
      # Increase counter + update target networks
      #############################################
      self.train_step_counter.assign_add(1)
      self._update_target()

    return tf_agent.LossInfo(loss=total_loss, extra=())


