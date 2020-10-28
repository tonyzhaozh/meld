from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys
import gin
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tf_agents.specs import tensor_spec

from meld.utils import nest_utils
from meld.networks.model_network_helpers import ConstantMultivariateNormalDiag, MultivariateNormalDiag
from meld.networks.model_network_helpers import ImageDecoder, RewardDecoder, BiggerRewardDecoder, ImageDecoderState
from meld.networks.model_network_helpers import ModelCompressor

@gin.configurable
class ModelDistributionNetwork(tf.Module):

  def __init__(self,

               # rollout info
               episodes_per_trial=None,
               max_episode_len=None,

               # network arch
               model_input='image', #image, state
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32), #None means learned
               reward_stddev=None, #None means learned, value means fixed
               double_camera=False, # used by compressor + image decoder to decide their architecture
              
               # training
               elbo_reward_weight=8,
               kl_analytic=True,

               # rewards
               model_reward=False,
               task_reward_dim=None, # dimension of rew from env's obs, always 1
               num_repeat_when_concatenate=None, # how many copies of r to concat to features
               reward_option='concat', # concat, outer_prod
               sparse_reward_inputs=None,
               sparse_reward_targets=None,

               # other
               length_of_vis_images=20, # used for visualization of images sampled from prior
               observation_spec=None,
               name=None,
               log_reconstruc_gifs=False, ## False by default, to save space/mem
               use_boundary_model=False,
               ):

    ########################
    # init vars
    ########################

    if num_repeat_when_concatenate is None:
      num_repeat_when_concatenate = base_depth

    super(ModelDistributionNetwork, self).__init__(name=name)
    self.sparse_reward_inputs = sparse_reward_inputs
    self.sparse_reward_targets = sparse_reward_targets
    self.reward_option=reward_option
    self.base_depth = base_depth
    self.latent1_size = latent1_size
    self.latent2_size = latent2_size
    self.kl_analytic = kl_analytic
    self.model_reward = model_reward
    self._num_repeat_when_concatenate = num_repeat_when_concatenate
    self.elbo_reward_weight = elbo_reward_weight
    self.model_input = model_input
    self.observation_spec = observation_spec
    self.task_reward_dim = task_reward_dim
    self.length_of_vis_images = length_of_vis_images
    self.episodes_per_trial = episodes_per_trial
    self.max_episode_len = max_episode_len
    self.log_reconstruc_gifs = log_reconstruc_gifs
    self.use_boundary_model = use_boundary_model

    # use this var to decide dense vs sparse rew for decoder target
    assert self.sparse_reward_targets is not None, 'please provide this argument explicitly'
    assert self.sparse_reward_inputs is not None, 'please provide this argument explicitly'
    # episode info
    assert self.episodes_per_trial is not None, 'please provide this argument explicitly'
    assert self.max_episode_len is not None, 'please provide this argument explicitly'

    # latent specs
    self.latent_dists_spec = tensor_spec.TensorSpec(shape=(2*latent1_size+2*latent2_size,), dtype=tf.float32, name='latent')
    self.latent_samples_spec = tensor_spec.TensorSpec(shape=(latent1_size+latent2_size,), dtype=tf.float32, name='latent')

    ########################
    # distributions to use
    ########################

    latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent1_distribution_ctor = MultivariateNormalDiag
    latent2_distribution_ctor = MultivariateNormalDiag

    ########################
    # priors
    ########################

    ##### 1st step

    # p(z_1^1)
    self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
    # p(z_1^2 | z_1^1)
    self.latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)

    ##### remaining steps

    # p(z_{t+1}^1 | z_t^2, a_t)
    self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    
    ##### boundary step

    if episodes_per_trial>1 and self.use_boundary_model:
      self.latent1_boundary_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
      self.latent2_boundary_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    else:
      self.latent1_boundary_prior = None
      self.latent2_boundary_prior = None 

    ########################
    # posteriors
    ########################

    ##### 1st step

    # q(z_1^1 | x_1)
    self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    self.latent2_first_posterior = self.latent2_first_prior

    ##### remaining steps

    # q(z_{t+1}^1 | x_{t+1}, z_{t}^2, a_{t}) 
    # NOTE THIS x_{t+1} for us should be [I_{t+1},r_{t}]
    self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_{t+1}^2 | z_{t+1}^1, z_{t}^2, a_{t}) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_posterior = self.latent2_prior
    
    ##### boundary step

    if episodes_per_trial>1 and self.use_boundary_model:
      # boundary posterior for z1, z2
      self.latent1_boundary_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
      self.latent2_boundary_posterior = self.latent2_boundary_prior
    else:
      self.latent1_boundary_posterior = None 
      self.latent2_boundary_posterior = None


    ########################
    # compressor + decoders
    ########################

    # compress images --> features
    self.compressor = ModelCompressor(base_depth, feature_size=8 * base_depth, double_camera=double_camera)

    # Image or state decoder
    # p(I_{t+1} | z_{t+1}^1, z_{t+1}^2)
    if self.model_input == 'image':
      self.decoder = ImageDecoder(base_depth, scale=decoder_stddev, double_camera=double_camera)
    elif self.model_input == 'state':
      state_size = observation_spec['state'].shape[0].value - 2 * self.task_reward_dim
      self.decoder = ImageDecoderState(base_depth, state_size=state_size, scale=decoder_stddev) # currently fixed scale

    # Reward decoder
    # p(r_{t} | z_{t+1}^1, z_{t+1}^2)
    assert self.model_reward, "MELD models reward"
    if self.model_reward:
      self.reward_decoder = RewardDecoder(8 * base_depth, scale=reward_stddev)
    else:
      self.reward_decoder = None


  #############################
  # dim of latents
  #############################

  @property
  def state_size(self):
    return self.latent1_size + self.latent2_size


  #############################
  # Utility functions
  #############################

  def decode_latents_into_reward(self, latent1_samples, latent2_samples):
    # latent1_samples: (num_latent_samples, 1, z1_dim)
    # latent2_samples: (num_latent_samples, 1, z2_dim)
    rew_dists = self.reward_decoder(latent1_samples, latent2_samples)
    return rew_dists.mean(), rew_dists.stddev()

  ### Augmented state is the concatenation of (state, step_within_episode, sparse_reward, dense_reward)

  def extract_sparse_dense_reward(self, augmented_state):
    assert self.task_reward_dim == 1
    task_reward_sparse = augmented_state[..., -self.task_reward_dim*2 : -self.task_reward_dim] # -2, sparse
    task_reward = augmented_state[..., -self.task_reward_dim:] # -1, dense
    return task_reward_sparse, task_reward

  def extract_state(self, augmented_state):
    assert self.task_reward_dim == 1
    state = augmented_state[..., :-self.task_reward_dim * 2 - 1]
    return state

  def extract_step_within_episode(self, augmented_state):
    assert self.task_reward_dim == 1
    step_within_episode = augmented_state[..., -self.task_reward_dim * 2 - 1]
    return step_within_episode

  def augment_features_with_rewards(self, features, rewards, option):
    assert option in ['concat', 'outerprod']

    if option == "concat":
      tiled_rew = tf.tile(rewards, [1, 1, self._num_repeat_when_concatenate])
      features = tf.concat([features, tiled_rew], axis=-1)

    elif option == "outerprod":
      # result will be [features*rewards, many1s*rewards, features*1, 1s]

      # (B,T,f) --> (B,T,2f) --> (B,T,1.5f) --> (B,T,1.5f,1)
      my_features = tf.concat([features, tf.ones_like(features)], 2)[:, :, :-features.shape[2].value // 3]
      my_features = tf.expand_dims(my_features, 3)

      # (B,T,1) --> (B,T,2) --> (B,T,1,2)
      my_task_reward = tf.concat([rewards, tf.ones_like(rewards)], 2)
      my_task_reward = tf.expand_dims(my_task_reward, 2)

      # outerprod = multiple (B,T,f,2) + flatten (B,T,f*2)
      features = tf.reshape(tf.multiply(my_features, my_task_reward),
                            (features.shape[0].value, features.shape[1].value, -1))

    return features


  #############################
  # loss for training this LVM
  #############################

  def concat_dist_params(self, dist):
    return tf.concat([dist.mean(), dist.stddev()], axis=-1)

  def compute_loss(self, images, augmented_states, actions, step_types, model_input):
    """
    inputs:
      (bs_in_num_trials, (episode_len+1)*episodes_per_trial, individual_shapes...)
    """

    ###########################
    # sample the posterior
    ###########################

    print("    Setting up posterior sampling")
    if model_input == 'image':
      latent_posterior_samples_and_dists = self.sample_prior_or_posterior(augmented_states=augmented_states,
                                                                          images=images, actions=actions,
                                                                          step_types=step_types,)

    elif model_input == 'state':
      latent_posterior_samples_and_dists = self.sample_prior_or_posterior(augmented_states=augmented_states,
                                                                          actions=actions, 
                                                                          step_types=step_types)

    else:
      raise NotImplementedError

    (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = latent_posterior_samples_and_dists

    ##############################################################
    # sample the prior, with prev timestep's POSTERIOR as input
    # used for the KL loss
    ##############################################################

    print("    Setting up prior dists for KL div")
    latent1_prior_dists, latent2_prior_dists = [], []
    for t in range(step_types.shape[1].value):
      # first step
      if t==0:
        # z1_0 ~ N(0,I)
        latent1_prior_dist = self.latent1_first_prior(step_types[:,t])  # step_types is only used to infer batch_size
        latent1_prior_sample = latent1_prior_dist.sample()

        # z2_0 ~ p(z2_0|z1_0)
        latent2_prior_dist = self.latent2_first_prior(latent1_posterior_samples[:, t])

      # boundary step
      elif ((t % (self.max_episode_len + 1) == 0) and self.use_boundary_model):
        # z1_curr~p(z1_curr|z2_POST_prev)
        latent1_prior_dist = self.latent1_boundary_prior(latent2_posterior_samples[:, t-1])
        latent1_prior_sample = latent1_prior_dist.sample()

        # z2_curr ~ p(z2_curr|z1_curr, z2_POST_prev)
        latent2_prior_dist = self.latent2_boundary_prior(latent1_posterior_samples[:, t], latent2_posterior_samples[:, t-1])

      # remaining steps
      else:
        # z1_curr~p(z1_curr|z2_POST_prev, a_prev)
        latent1_prior_dist = self.latent1_prior(latent2_posterior_samples[:, t-1], actions[:, t-1])
        latent1_prior_sample = latent1_prior_dist.sample()

        # z2_curr ~ p(z2_curr|z1_curr, z2_POST_prev, a_prev)
        latent2_prior_dist = self.latent2_prior(latent1_posterior_samples[:, t], latent2_posterior_samples[:, t-1], actions[:, t-1])

      latent1_prior_dists.append(latent1_prior_dist)
      latent2_prior_dists.append(latent2_prior_dist)

    latent1_prior_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_prior_dists)
    latent2_prior_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_prior_dists)

    ###########################
    # sample conditional prior 
    # used only for visualization
    ###########################

    print("    Setting up prior/cond sampling (on shorter sequences) for visualization")

    # Sample the prior (doesn't need rewards/images/etc.)
    (latent1_prior_samples, latent2_prior_samples), _ = self.sample_prior_or_posterior(actions=actions[:, :self.length_of_vis_images],
                                                                                      step_types=step_types[:, :self.length_of_vis_images])

    # Sample the conditional prior, conditioned on only 1st image of the sequence
    if model_input == 'image':
      (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_prior_or_posterior(
        actions[:, :self.length_of_vis_images],
        step_types=step_types[:, :self.length_of_vis_images],
        images=images[:, :1], # only first image given
        augmented_states=augmented_states[:, :1])
    elif model_input == 'state':
      (latent1_conditional_prior_samples, latent2_conditional_prior_samples), _ = self.sample_prior_or_posterior(
        actions[:, :self.length_of_vis_images],
        step_types=step_types[:, :self.length_of_vis_images],
        augmented_states=augmented_states[:, :1]) # only first state given
    else:
      raise NotImplementedError


    ##################################
    # KL div (prior vs posterior)
    ##################################

    ### z1

    print("    Setting up elbo/all model training losses")
    outputs = {}
    if self.kl_analytic:
      latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
    else:
      latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples) - latent1_prior_dists.log_prob(latent1_posterior_samples))
    latent1_kl_divergences = tf.reduce_mean(latent1_kl_divergences, axis=1)
    outputs.update({'latent1_kl_divergence': tf.reduce_mean(latent1_kl_divergences),})

    ### z2

    latent2_kl_divergences = 0.0 # only constrain

    ### log sum of layer1 and layer2 KL
    outputs.update({'kl_divergence': tf.reduce_mean(latent1_kl_divergences + latent2_kl_divergences),})
    

    ##################################
    # reconstruct images/state
    # p(I_t|z_t)
    ##################################

    if model_input == 'image':
      reconstruct_target = images
    elif model_input == 'state':
      reconstruct_target = self.extract_state(augmented_states)
    else:
      raise NotImplementedError

    # decode the posterior samples
    likelihood_dists = self.decoder(latent1_posterior_samples, latent2_posterior_samples)
    likelihood_log_probs = likelihood_dists.log_prob(reconstruct_target)
    likelihood_log_probs = tf.reduce_mean(likelihood_log_probs, axis=1)
    reconstruction_error = tf.reduce_mean(tf.square(reconstruct_target - likelihood_dists.distribution.loc),
                                         axis=list(range(-len(likelihood_dists.event_shape), 0)))
    reconstruction_error = tf.reduce_mean(reconstruction_error, axis=1)
    outputs.update({
      'log_likelihood': tf.reduce_mean(likelihood_log_probs),
      'reconstruction_error': tf.reduce_mean(reconstruction_error),
    })



    ###########################
    # reconstruct (dense) rewards
    # p(r_t|z_t)
    ###########################

    task_reward_sparse, task_reward = self.extract_sparse_dense_reward(augmented_states)
    target_task_reward = task_reward_sparse if self.sparse_reward_targets else task_reward

    reconst_reward_log_probs = 0
    reconst_reward_log_probs2 = 0
    if self.model_reward:

      # decode the posterior samples
      reconst_reward_dists = self.reward_decoder(latent1_posterior_samples, latent2_posterior_samples)
      reconst_reward_log_probs = reconst_reward_dists.log_prob(tf.squeeze(target_task_reward))
      reconst_reward_log_probs = tf.reduce_mean(reconst_reward_log_probs, axis=1)

      reconst_reward_error = tf.sqrt(tf.square(tf.squeeze(target_task_reward) - reconst_reward_dists.loc))
      reconst_reward_error = tf.reduce_mean(reconst_reward_error, axis=1)

      outputs.update({
        'reconst_reward_log_likelihood': tf.reduce_mean(reconst_reward_log_probs),
        'reconst_reward_error': tf.reduce_mean(reconst_reward_error), # when logging, use abs value
      })


    ########################
    # total elbo + logging
    ########################

    elbo = self.elbo_normal_training(likelihood_log_probs, latent1_kl_divergences, latent2_kl_divergences, reconst_reward_log_probs)

    loss = -tf.reduce_mean(elbo) # average over the batch dimension
    outputs.update({'elbo': tf.reduce_mean(elbo),})

    # for visualization(logging to tensorboard)
    if model_input == 'image':
      posterior_images = likelihood_dists.mean()
      prior_images = self.decoder(latent1_prior_samples, latent2_prior_samples).mean()
      conditional_prior_images = self.decoder(latent1_conditional_prior_samples, latent2_conditional_prior_samples).mean()

      if self.log_reconstruc_gifs:
        outputs.update({
          'images': images,
          'posterior_images': posterior_images,
          'prior_images': prior_images,
          'conditional_prior_images': conditional_prior_images,
        })

    return loss, outputs


  def elbo_normal_training(self, likelihood_log_probs, latent1_kl_divergences, latent2_kl_divergences,
                           reconst_reward_log_probs):
    elbo = likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences + reconst_reward_log_probs*self.elbo_reward_weight
    return elbo



  #################################################################
  # sample latents from posterior (if images present), else from prior
  # function to be used during training (i.e. fixed seq length, set to whatever it was created with)
  #################################################################

  def get_features(self, images, augmented_states):

    # images/state --> compressor --> features
    if self.model_input == 'image':
      features = self.compressor(images)  # (15, 25, 256)
    elif self.model_input == 'state':
      features = self.extract_state(augmented_states)
    else:
      raise NotImplementedError

    # get sparse/dense rewards
    task_reward_sparse, task_reward_dense = self.extract_sparse_dense_reward(augmented_states)

    # concat the image features with the rews: [features, rewards]
    features_dense = self.augment_features_with_rewards(features, task_reward_dense, self.reward_option)
    features_sparse = self.augment_features_with_rewards(features, task_reward_sparse, self.reward_option)

    return features_dense, features_sparse

  def sample_prior_or_posterior(self, actions, step_types, images=None, augmented_states=None,
                                randomly_select_posterior=False, which_posterior='first'):

    ###############################################
    # Prepare actions, step_types, and features
    ###############################################

    sequence_length = step_types.shape[1].value - 1 # last transition in a trial is not meaningful
    actions = actions[:, :sequence_length] # discard the last action, since it is in "between trial boundary"
    step_types = step_types[:, :sequence_length+1]

    # Get [features, rewards]
    features = None
    latent1_first_posterior_use = None
    if not images is None:

      features_dense, features_sparse = self.get_features(images, augmented_states)

      # select the features based on reward option and/or which posterior we are sampling from
      features = features_sparse if self.sparse_reward_inputs else features_dense

    # sample from 1st (and potentially only) posterior
    if latent1_first_posterior_use is None:
      latent1_first_posterior_use = self.latent1_first_posterior
      latent2_first_posterior_use = self.latent2_first_posterior
      latent1_boundary_posterior_use = self.latent1_boundary_posterior
      latent2_boundary_posterior_use = self.latent2_boundary_posterior
      latent1_posterior_use = self.latent1_posterior
      latent2_posterior_use = self.latent2_posterior

    # swap batch and time axes for actions and step_types
    actions = tf.transpose(actions, [1, 0, 2])
    step_types = tf.transpose(step_types, [1, 0])
    if features is not None:
      features = tf.transpose(features, [1, 0, 2])

    """
    Dims, at this point

    images: [bs_in_num_trials, (episode_len+1)*episodes_per_trial, 64, 64, 3]

    actions: [(episode_len+1)*episodes_per_trial-1, bs_in_num_trials, acdim]
    step_types: [(episode_len+1)*episodes_per_trial, bs_in_num_trials]
    features: [(episode_len+1)*episodes_per_trial, bs_in_num_trials, self.compressor.feature_size + self._num_repeat_when_concatenate]
    """

    ############################
    # Sample latents
    ############################

    print("        Created model sampling procedure with seq length ", sequence_length)
    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []

    bs = step_types[0].shape[0].value
    prev_latents = tf.zeros((bs, self.latent1_size+self.latent2_size))

    for t in range(sequence_length + 1):

      # existence of features decides prior vs posterior
      is_conditional = (features is not None) and (t < features.shape[0].value)

      # step within episode (0 to max_episode_len)
      # for t ranging from 0 to max_episode_len*episodes_per_trial
      step_within_episode = t%(self.max_episode_len+1)

      ###########################
      # first step in a trial
      ###########################

      if t==0:
        # sample latent 1
        if is_conditional:
          latent1_dist = latent1_first_posterior_use(features[t])
        else:
          latent1_dist = self.latent1_first_prior(step_types[t])  # step_types used to infer batch_size
        latent1_sample = latent1_dist.sample()

        # sample latent 2
        if is_conditional:
          latent2_dist = latent2_first_posterior_use(latent1_sample)
        else:
          latent2_dist = self.latent2_first_prior(latent1_sample)
        latent2_sample = latent2_dist.sample()

      ###########################
      # boundary steps inside a trial
      ###########################

      elif ((t % (self.max_episode_len+1) == 0) and self.use_boundary_model):
        # if conditional, z1_curr~posterior(z1_curr|I_curr, z2_prev, a_prev)
        # else, z1_curr~prior(z1_curr|z2_prev, a_prev)
        if is_conditional:
          latent1_dist = latent1_boundary_posterior_use(features[t], latent2_samples[t-1])
        else:
          latent1_dist = self.latent1_boundary_prior(latent2_samples[t-1])
        latent1_sample = latent1_dist.sample()

        # if conditional, z2_curr~posterior(z2_curr|z1_curr, z2_prev, a_prev)
            # note: this z1_curr would be posterior from above, so it saw image
        # else, z2_curr~prior(z2_curr|z1_curr, z2_prev, a_prev)
            # note: this z1_curr would be prior from above
        if is_conditional:
          latent2_dist = latent2_boundary_posterior_use(latent1_sample, latent2_samples[t-1])
        else:
          latent2_dist = self.latent2_boundary_prior(latent1_sample, latent2_samples[t-1])
        latent2_sample = latent2_dist.sample()

      ###########################
      # normal steps in a trial
      ###########################

      else:
        # if conditional, z1_curr~posterior(z1_curr|I_curr, z2_prev, a_prev)
        # else, z1_curr~prior(z1_curr|z2_prev, a_prev)
        if is_conditional:
          latent1_dist = latent1_posterior_use(features[t], latent2_samples[t-1], actions[t-1])
        else:
          latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
        latent1_sample = latent1_dist.sample()

        # if conditional, z2_curr~posterior(z2_curr|z1_curr, z2_prev, a_prev)
            # note: this z1_curr would be posterior from above, so it saw image
        # else, z2_curr~prior(z2_curr|z1_curr, z2_prev, a_prev)
            # note: this z1_curr would be prior from above
        if is_conditional:
          latent2_dist = latent2_posterior_use(latent1_sample, latent2_samples[t-1], actions[t-1])
        else:
          latent2_dist = self.latent2_prior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist); latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist); latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)
    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)


  ###########################################################
  # function to be used by policy (always single-step history)
  ###########################################################


  def sample_onestep_forward(self, image, augmented_state, prev_latentsOrDists, action_taken, 
                            return_distribution=False, which_posterior='first', which_rew_input='dense'):
    """
    Called by POLICY, with new observations from the env
      Called during data collection rollouts + eval rollouts
      Note that it's not called during train steps, because that just queries actor_network directly without needing any z calcs
    
    ########################

    Function: (image-->compressor-->features) --> (features,rewards), prev_ac, prev_z --> sample posterior --> z
    
    :param image: the new image observation (from env)
    :param augmented_state: state/sparserew/denserew (from env)
    :param prev_latentsOrDists: z1,z2 sample from previous latent distribution OR z1mean,z1std,z2mean,z2std
    :param action_taken: the prev action taken by policy (conditioned on prev_latentsOrDists) to arrive at current state

    :param return_distribution:
    if True: return dists, samples
    else: return samples

    where
      dists = [latent1_dist.mean(), latent1_dist.stddev(),latent2_dist.mean(), latent2_dist.stddev()]
      samples = [latent1_sample,latent2_sample]
    """

    # sanity checks
    assert self.model_reward
    assert self.model_input in ["image", "state"]

    ######################################
    # image/state --> compressor --> features
    ######################################

    features = None
    if self.model_input == 'image':
      features = self.compressor(image)
    elif self.model_input == 'state':
      features = self.extract_state(augmented_state)
    else:
      raise NotImplementedError

    ######################################
    # features --> (features, rew)
    ######################################

    # rew_inp_print = tf.print("REWARD INPUT IS: ", which_rew_input, output_stream=sys.stdout)
    # with tf.control_dependencies([rew_inp_print]):
    task_reward_sparse, task_reward_dense = self.extract_sparse_dense_reward(augmented_state)

    if which_rew_input=='sparse':
      input_reward = task_reward_sparse
    elif which_rew_input=='dense':
      input_reward = task_reward_dense
    else:
      raise NotImplementedError
    features = self.augment_features_with_rewards(features, input_reward, self.reward_option)

    """
    Dims, at this point
    action_taken: [1, 1, action_dim]
    features: [1, 1, self.compressor.feature_size + self._num_repeat_when_concatenate]
    prev_latentsOrDists: [1, 1, self.latent1_size + self.latent2_size] or [1, 1, 2*self.latent1_size + 2*self.latent2_size]
    """

    ######################################
    # which step of rollout
    ######################################

    # Get step within episode (from augmented state)
    step_within_episode = self.extract_step_within_episode(augmented_state)
    step_within_episode = tf.squeeze(step_within_episode)

    #####################################################
    # ([feat,rew], prev_ac, prev_z) --> posterior --> z
    #####################################################

    """
    if prev_latentsOrDists is all zeros, 
      this is beginning of a trial
      sample from 1st-step 
    elif step_within_episode is 0,
      this is beginning of a rollout (but not beginning of a trial)
      sample from boundary-step
    else,
      this is a normal step so sample from normal
    """

    if self.episodes_per_trial>1 and self.use_boundary_model:
      latents = tf.cond(tf.equal((tf.count_nonzero(prev_latentsOrDists[0])), 0), # start of trial
                             lambda: self.samp_posterior_first_step(features[0], return_distribution=return_distribution, step_within_episode=step_within_episode, which_posterior=which_posterior),
                             lambda: tf.cond(tf.equal(step_within_episode, 0), # between episodes within a trial
                                             lambda: self.samp_posterior_boundary_step(features[0], prev_latentsOrDists[0], return_distribution=return_distribution, step_within_episode=step_within_episode, which_posterior=which_posterior),
                                             lambda: self.samp_posterior_normal_step(features[0], prev_latentsOrDists[0], action_taken[0], return_distribution=return_distribution, step_within_episode=step_within_episode, which_posterior=which_posterior),))
    else:
      latents = tf.cond(tf.equal((tf.count_nonzero(prev_latentsOrDists[0])), 0), # start of trial
                             lambda: self.samp_posterior_first_step(features[0], return_distribution=return_distribution, step_within_episode=step_within_episode, which_posterior=which_posterior),
                             lambda: self.samp_posterior_normal_step(features[0], prev_latentsOrDists[0], action_taken[0], return_distribution=return_distribution, step_within_episode=step_within_episode, which_posterior=which_posterior))

    return latents


  #####################################################
  # helpers for posterior sampling
  #####################################################


  def samp_posterior_first_step(self, features, return_distribution=False, step_within_episode=None, which_posterior=None):

    # check step number
    # turned this off for batchmode version
    # tf.debugging.assert_equal(step_within_episode, tf.constant(0, dtype=step_within_episode.dtype), message="need to be zero, obviously")

    if which_posterior=='first':
      latent1_first_posterior_use = self.latent1_first_posterior
      latent2_first_posterior_use = self.latent2_first_posterior
    elif which_posterior=='second':
      latent1_first_posterior_use = self.latent1_first_posterior2
      latent2_first_posterior_use = self.latent2_first_posterior2
    else:
      raise NotImplementedError

    #############

    sample_first_posterior_print = tf.print("~~ Sampling latent for policy (for t=0) from FIRST posterior!", output_stream=sys.stdout)
    with tf.control_dependencies([sample_first_posterior_print]):
      # z1_0 ~ p(z1_0|I_0)
      latent1_dist = latent1_first_posterior_use(features)
      latent1_sample = latent1_dist.sample()

      # z2_0 ~ p(z2_0|z1_0)
      latent2_dist = latent2_first_posterior_use(latent1_sample)
      latent2_sample = latent2_dist.sample()

    if return_distribution:
      return [latent1_dist.mean(), latent1_dist.stddev(),latent2_dist.mean(), latent2_dist.stddev()], [latent1_sample,latent2_sample]
    else:
      return [latent1_sample, latent2_sample]


  def samp_posterior_normal_step(self, features, prev_latentsOrDists, action_to_take, return_distribution=False, step_within_episode=None, which_posterior=None):

    if which_posterior=='first':
      latent1_posterior_use = self.latent1_posterior
      latent2_posterior_use = self.latent2_posterior
    elif which_posterior=='second':
      latent1_posterior_use = self.latent1_posterior2
      latent2_posterior_use = self.latent2_posterior2
    else:
      raise NotImplementedError

    #############

    # get just previous z2 sample or dist
    # (1,lat1+lat2) --> (1,lat2)
    # [z2sample]
    prev_latent2 = prev_latentsOrDists[:,self.latent1_size:]
    
    # z1_curr~posterior(z1_curr|I_curr, z2_prev, a_prev)
    latent1_dist = latent1_posterior_use(features, prev_latent2, action_to_take)
    latent1_sample = latent1_dist.sample()

    # z2_curr~posterior(z2_curr|z1_curr, z2_prev, a_prev)
    latent2_dist = latent2_posterior_use(latent1_sample, prev_latent2, action_to_take)
    latent2_sample = latent2_dist.sample()

    if return_distribution:
      return [latent1_dist.mean(), latent1_dist.stddev(),latent2_dist.mean(), latent2_dist.stddev()], [latent1_sample, latent2_sample]
    else:
      return [latent1_sample,latent2_sample]


  def samp_posterior_boundary_step(self, features, prev_latentsOrDists, return_distribution=False, step_within_episode=None, which_posterior=None):

    if which_posterior=='first':
      latent1_boundary_posterior_use = self.latent1_boundary_posterior
      latent2_boundary_posterior_use = self.latent2_boundary_posterior
    elif which_posterior=='second':
      latent1_boundary_posterior_use = self.latent1_boundary_posterior2
      latent2_boundary_posterior_use = self.latent2_boundary_posterior2
    else:
      raise NotImplementedError

    #############

    sample_boundary_print = tf.print("~~ Sampling latent for policy (for t=0) from BOUNDARY posterior!", output_stream=sys.stdout)
    with tf.control_dependencies([sample_boundary_print]):

      # get just previous z2 sample or dist
      prev_latent2 = prev_latentsOrDists[:, self.latent1_size:]

      # z1_curr~posterior(z1_curr|I_curr, z2_prev)
      latent1_dist = latent1_boundary_posterior_use(features, prev_latent2,) # notice: do not take in action
      latent1_sample = latent1_dist.sample()

      # z2_curr~posterior(z2_curr|z1_curr, z2_prev)
      latent2_dist = latent2_boundary_posterior_use(latent1_sample, prev_latent2,)
      latent2_sample = latent2_dist.sample()

    if return_distribution:
      return [latent1_dist.mean(), latent1_dist.stddev(), latent2_dist.mean(), latent2_dist.stddev()], [latent1_sample, latent2_sample]
    else:
      return [latent1_sample, latent2_sample]