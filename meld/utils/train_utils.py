from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf


################################
# Actor Loss
################################

def actor_loss_fn(observation, step_type, actor_network, 
                      critic_network1, critic_network2, 
                      log_alpha, actor_input_stop_gradient, 
                      weights=None):

  """
    actorloss(s_t) = - Q(s_t, pi(a_t)) + ent(pi)
  """

  with tf.name_scope('actor_loss'):

    #######################
    # stop gradients
    #######################

    # IMPT NOTE: actor_input_stop_gradient stops gradient
    # for both actor and critic

    if actor_input_stop_gradient:
      observation, step_type = tf.nest.map_structure(tf.stop_gradient, [observation, step_type])

    #######################
    # a_t ~ pi(s_t)
    #######################

    actions_distribution, _ = actor_network(observation, step_type)
    actions = actions_distribution.sample()
    log_pis = actions_distribution.log_prob(actions)

    #######################
    # Q(s_t, a_t)
    #######################

    target_input_1 = (observation, actions)
    target_q_values1, unused_network_state1 = critic_network1(target_input_1, step_type)
    target_input_2 = (observation, actions)
    target_q_values2, unused_network_state2 = critic_network2(target_input_2, step_type)
    target_q_values = tf.minimum(target_q_values1, target_q_values2)

    #######################
    # loss = - [Q(s_t, a_t) + ent]
    #######################

    actor_loss = tf.exp(log_alpha) * log_pis - target_q_values
    if weights is not None:
      actor_loss *= weights
    actor_loss = tf.reduce_mean(input_tensor=actor_loss)

    return actor_loss


################################
# Critic Loss
################################

def critic_loss_fn(observation, next_observation, 
                        step_type, next_step_type, 
                        next_reward, next_discount, actions,
                        td_errors_loss_fn, critic_input_stop_gradient, log_alpha,
                        actor_network, target_critic_network1, target_critic_network2,
                        critic_network1, critic_network2,
                        gamma=0.99, reward_scale_factor=1.0, weights=None):

  """
    criticloss(s_t-1, a_t-1, s_t) = ||Q(s_t-1, a_t-1) - (rew + g*discount*(Q(s_t,pi(s_t)) + ent))||
  """

  with tf.name_scope('critic_loss'):

    #######################
    # stop gradients
    #######################

    if critic_input_stop_gradient:
      observation, next_observation, step_type, next_step_type, next_reward, next_discount, actions = \
        tf.nest.map_structure(tf.stop_gradient,
                              [observation, next_observation, step_type, next_step_type, next_reward, next_discount, actions])


    # if critic_input_stop_gradient:
    #   time_steps = tf.nest.map_structure(tf.stop_gradient, time_steps)
    #   next_time_steps = tf.nest.map_structure(tf.stop_gradient, next_time_steps)
    # actor_next_time_steps = tf.nest.map_structure(tf.stop_gradient, actor_next_time_steps)

    #######################
    # a_t ~ pi(s_t)
    #######################

    next_actions_distribution, _ = actor_network(next_observation) # was actor distribution
    next_actions = next_actions_distribution.sample()
    next_log_pis = next_actions_distribution.log_prob(next_actions)

    #######################
    # Q(s_t,a_t) + entropy
    #######################

    target_input_1 = (next_observation, next_actions)
    target_q_values1, unused_network_state1 = target_critic_network1(target_input_1, next_step_type)
    target_input_2 = (next_observation, next_actions)
    target_q_values2, unused_network_state2 = target_critic_network2(target_input_2, next_step_type)
    target_q_values = (
        tf.minimum(target_q_values1, target_q_values2) -
        tf.exp(log_alpha) * next_log_pis)

    #######################
    # target_t = r + gamma*[Q(s_t,a_t)+ent]
    #######################

    td_targets = tf.stop_gradient(
        reward_scale_factor * next_reward + # this rew is result of (z_t,a_t)
        gamma * next_discount * target_q_values)

    #######################
    # loss = ||Q(s_t-1, a_t-1) - target_t||
    #######################

    pred_input_1 = (observation, actions)
    pred_td_targets1, unused_network_state1 = critic_network1(pred_input_1, step_type)
    pred_input_2 = (observation, actions)
    pred_td_targets2, unused_network_state2 = critic_network2(pred_input_2, step_type)
    critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
    critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
    critic_loss = critic_loss1 + critic_loss2

    #######################
    # mean across batch
    #######################

    if weights is not None:
      critic_loss *= weights
    critic_loss = tf.reduce_mean(input_tensor=critic_loss)
    return critic_loss


################################
# Model Loss
################################

def model_loss_fn(images, augmented_states, actions, step_types, model_input, model_network, weights=None):

  with tf.name_scope('model_loss'):

    #####################################
    # compute loss (elbo for LVM)
    #####################################
    model_loss, outputs = model_network.compute_loss(images, augmented_states, actions, step_types, model_input)

    #####################################
    # return
    #####################################
    if weights is not None:
      model_loss *= weights
    model_loss = tf.reduce_mean(input_tensor=model_loss)
    return model_loss, outputs


################################
# Alpha Loss
################################

def alpha_loss_fn(observation, step_type,
                       actor_network, log_alpha, 
                       target_entropy,
                       weights=None):

  with tf.name_scope('alpha_loss'):
    actions_distribution, _ = actor_network(observation, step_type)
    actions = actions_distribution.sample()
    log_pis = actions_distribution.log_prob(actions)
    alpha_loss = (log_alpha * tf.stop_gradient(-log_pis - target_entropy))
    if weights is not None:
      alpha_loss *= weights
    alpha_loss = tf.reduce_mean(input_tensor=alpha_loss)
    return alpha_loss
