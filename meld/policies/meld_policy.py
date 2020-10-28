import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.networks import network


class MeldPolicy(tf_policy.Base):

  """
  Custom policy class for MELD.
  
  What happens when policy is queried:
    ([img_curr, rew_curr], a_prev, z_prev_samples, z_prev_dists) → model_network → z → actor_network → action

  What is maintained internally in policy_state:
    policy_state=[a_prev, z_prev_samples, z_prev_dists]

  When policy_state gets reset:
    `get_initial_policy_state` clears policy_state to zeros
    whenever policy gets a timestep where `time_step.is_first()`

  Note: this policy is used as "policy" for evaluation rollouts, --> this should always eval with sparse
  or as "collection_policy" for collecting data during the training process. --> this can just always collect data with dense
  """

  def __init__(self,
               # specs
               time_step_spec=None,
               action_spec=None,
               info_spec=(),
               # networks
               actor_network=None,
               model_network=None,
               # input type for actor network
               actor_input='latentSample',
               name=None,
               which_posterior='first', #first, second
               which_rew_input='dense', #dense, sparse
               ):

    #################################
    # init vars
    #################################

    # check actor_network
    if not isinstance(actor_network, network.Network):
      raise ValueError('actor_network must be a network.Network. Found '
                       '{}.'.format(type(actor_network)))

    # networks
    self._actor_network = actor_network
    self._model_network = model_network
    self.which_posterior = which_posterior
    self.which_rew_input = which_rew_input

    # input type for actor network
    self._actor_input = actor_input
    assert (self._actor_input=='latentSample' or self._actor_input=='latentDistribution')

    #################################
    # specs
    #################################

    def _add_time_dimension(spec):
      return tensor_spec.TensorSpec(
          (1,) + tuple(spec.shape), spec.dtype, spec.name)

    policy_input_spec_samples = self._model_network.latent_samples_spec # (z1+z2,)
    policy_input_spec_samples_with_time = tf.nest.map_structure(_add_time_dimension, policy_input_spec_samples) #(x) -> (1,x)
    policy_input_spec_dists = self._model_network.latent_dists_spec # (z1mean+z1std+z2mean+z2std,)
    policy_input_spec_dists_with_time = tf.nest.map_structure(_add_time_dimension, policy_input_spec_dists) #(x) -> (1,x)
    policy_action_spec_with_time = tf.nest.map_structure(_add_time_dimension, action_spec) #(ac_dim) -> (1,ac_dim)

    # policy state maintains information including
    # (a) action from previous step (ie what to take now)
    # (b) previous latent samples
    # (c) previous latent dists
    policy_state_spec = (policy_action_spec_with_time, policy_input_spec_samples_with_time, policy_input_spec_dists_with_time) 

    ########################################
    # create tf agents policy with these specs
    ########################################

    super(MeldPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        name=name,
        # automatic_state_reset is True by default : `get_initial_policy_state` clears policy_state whenever policy gets timestep where `time_step.is_first()`
        )

    """
    Note: 
    The dimension we added above is to represent time: 
        (timesteps=1, latentdim) and (timesteps=1, ac_dim)
    Calling get_initial_state(bs) from external code automatically adds batch dim to these: 
        (bs,timesteps=1,latentdim) and (bs,timesteps=1,ac_dim)
    """
    self.action_spec = action_spec
    self.time_step_spec = time_step_spec
    self.policy_action_spec_with_time = policy_action_spec_with_time
    self.policy_input_spec_samples_with_time = policy_input_spec_samples_with_time
    self.policy_input_spec_dists_with_time = policy_input_spec_dists_with_time
    self.info_spec = info_spec
  
  ############################################################
  ############################################################

  def _variables(self):

    # actor network vars
    variables = list(self._actor_network.variables)

    # model network vars (since images go through there before going into actor)
    if 'latent' in self._actor_input:
      variables += self._model_network.variables

    return variables

  ############################################################
  ############################################################

  def _action(self, time_step, policy_state, seed):

    """
    This function is called by policy's action function.
    Called as policy.action(time_step, policy_state).
    Returns entry with entry.action and entry.state
    """

    # call _distribution below
    # which queries actor network to give us a dist to sample an action from
    # note: it also updates the latent stored in policy_state
    distribution_step = self.distribution(time_step, policy_state)

    # sample dist to get action for the env to take
    action = distribution_step.action.sample(seed=seed) 

    # update policy state's stored action
    # because this action is what should be passed into LVM at NEXT step
    policy_state = distribution_step.state
    _, prev_latent_samples, prev_latent_dists = policy_state
    policy_state = action[:,None], prev_latent_samples, prev_latent_dists # add a time dim to the ac

    # return this action to take
    # return the updated policy_state
    return distribution_step._replace(action=action, state=policy_state)

  ############################################################
  ############################################################

  def _distribution(self, time_step, policy_state):

    """
    This function is called by _action above. It does the following:
    (1) queries actor network to get action dist
    (2) updates the latent samples/dists stored in policy_state
    """

    # get the stored into
    prev_action, prev_latent_samples, prev_latent_dists = policy_state

    # query actor network (with img, rew, prev_ac, prev_latent)
    action_or_distribution, latent_samples, latent_dists = self._apply_actor_network(time_step, prev_action, prev_latent_samples, prev_latent_dists)

    # update policy state's stored latent (to be used at the next timestep) 
    policy_state = prev_action, latent_samples[:,None], latent_dists[:,None] # add a time dim before adding

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distribution = tf.nest.map_structure(_to_distribution, action_or_distribution)
    return policy_step.PolicyStep(distribution, policy_state)

  ############################################################
  ############################################################

  def _apply_actor_network(self, time_step, prev_action, prev_latent_samples, prev_latent_dists):

    # add batch dim
    obs_states = tf.expand_dims(time_step.observation['state'], axis=0) # (1,state_dim) based on time_step_spec --> (bs, 1, state_dim)
    obs_pixels = tf.expand_dims(time_step.observation['pixels'], axis=0) # (1,img_size,img_size,3) --> (bs,1,img_size,img_size,3)

    # pixels --> image
    image = tf.image.convert_image_dtype(obs_pixels, tf.float32)

    # inputs to the model
    prev_latent_sampleOrDist = prev_latent_samples

    # ([I_curr, r_curr], a_prev) --> model --> z_dist, z_sample
    latents_dist, latents_sample = self._model_network.sample_onestep_forward(image=image,
                                                                              augmented_state=obs_states, #contains sparse and dense reward inside
                                                                              prev_latentsOrDists=prev_latent_sampleOrDist, #(1,latdim)
                                                                              action_taken=prev_action,
                                                                              return_distribution=True,
                                                                              which_posterior=self.which_posterior,
                                                                              which_rew_input=self.which_rew_input,
                                                                              )
    latents_sample = tf.concat(latents_sample, axis=-1)  # concat [z1_sample, z2_sample]
    latents_dist = tf.concat(latents_dist, axis=-1)  # concat [z1.mean(), z1.stddev(), z2.mean(), z2.stddev()]

    # what to pass into actor network
    if self._actor_input == 'latentSample':
      actor_obs = latents_sample
    elif self._actor_input == 'latentDistribution':
      actor_obs = latents_dist
    else:
      raise NotImplementedError

    # query actor network for action dist
    dist, _ = self._actor_network(actor_obs)

    #latents_sample [z1_sample, z2_sample]
    #latents_dist [z1.mean(), z1.stddev(), z2.mean(), z2.stddev()]
    return dist, latents_sample, latents_dist