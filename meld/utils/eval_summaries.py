from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tf_agents.metrics import py_metric
from tf_agents.trajectories import trajectory

from meld.utils.utils import mask_episode_transition, pad_and_concatenate_videos

#######################################################
#######################################################


def perform_eval_and_summaries_meld(
                      eval_metrics,
                      env,
                      policy,
                      num_trials,
                      max_episode_len,
                      episodes_per_trial,
                      log_image_strips=False,
                      num_trials_to_render=1,
                      num_trials_to_render_overwrite=None,
                      eval_tasks=None,
                      model_input='image',
                      latent1_size=None,
                      latent2_size=None,
                      logger=None,
                      log_image_observations=False,
                      global_step_val=0,
                      render_fps=10,
                      decode_rews_op=None,
                      latent_samples_1_ph=None,
                      latent_samples_2_ph=None,
                      eval_doing_second_agent=False,
                      dont_plot_reward_gifs=True, ## dont plot, be default
                      ): 
  """
  Collect num_trials trials on env, using policy
  Where each trial is episodes_per_trial episodes
  And each episode is of length max_episode_len
  Render all episodes from num_trials_to_render trials

  NOTE: At the beginning of each trial,
  We reset the latent variable,
  and reset the task randomly from eval_tasks.
  """
  
  ###########
  # init
  ###########

  session = tf.compat.v1.get_default_session()

  if episodes_per_trial>1:
    num_trials_to_render = 1 #this is what will look good, because we visualize each episode from a single trial

  if num_trials_to_render_overwrite is not None:
    num_trials_to_render = num_trials_to_render_overwrite #can overwrite this during offline eval

  num_metrics_per_episode = int(len(eval_metrics) / episodes_per_trial)

  for metric in eval_metrics:
    metric.reset()
  rendering = False
  if num_trials_to_render:
    rendering = True
    env.start_rendering()

  #####################
  # select random tasks
  #####################

  if num_trials <= len(eval_tasks):
    selected_tasks_indices = np.random.choice(len(eval_tasks), num_trials, replace=False)
  else:
    selected_tasks_indices = np.random.choice(len(eval_tasks), num_trials, replace=True)

  #####################
  # reset env 
  #####################

  time_step = env.reset()

  #####################
  # init images holder
  #####################

  render_images = []
  if num_trials_to_render and 'pixels' in time_step.observation:
    images = [[]]
    variance_plots = []
  else:
    images = []
    variance_plots = []
  fig_list = []

  #####################
  # collect rollouts
  #####################

  for trial, task_idx in zip(range(num_trials), selected_tasks_indices):

    # set the task at beginning of each trial
    env.set_task_for_env(eval_tasks[task_idx])
    env.frames[:] = []  # clear buffer
    time_step = env.reset()

    # explicitly get the "reset" policy_state to feed into policy
    # to serve as a manual reset of the latent
    policy_state = policy.get_initial_state(env.batch_size)

    if rendering:
      if 'pixels' in time_step.observation:
        # print("BEGINNING OF TRIAL... ADDED...")
        images[-1].append(time_step.observation['pixels']) # first frame in a trial
    
    for episode in range(episodes_per_trial):
      # print('  episode ', episode, '/', episodes_per_trial)

      # init for plotting
      z1_stds, z2_stds, z1_means, z2_means = [], [], [], []
      rewards_dense, rewards_sparse = [], [] 
      mean_of_means, std_of_means, mean_of_vars, std_of_vars = [], [], [], []
      

      for step in range(max_episode_len):

        # print('      step ', step, '/', max_episode_len, ' steptype: ', time_step.step_type)

        # at beginning of each rollout, prevent latent from resetting
        # except for episode==0 (first episode of a trial)
        if step==0 and episode>0:
            # policy shall not see step type as first
            # thus, it will not reset the latent in its policy_state
            policy_time_step = mask_episode_transition(time_step) 
            # print("    I AM HIDING THE STEP TYPE..")
        else:
          policy_time_step = time_step

        # get action
        # get action(prev latent info)
        # note that action_step has .state which is curr latent info
        action_step = policy.action(policy_time_step, policy_state)

        # take step
        next_time_step = env.step(action_step.action)

        # plot info as gifs in tb summaries
        if rendering:
          if not dont_plot_reward_gifs:
            ######### rewards
            rewards_dense.append(next_time_step.observation['state'][-1])
            rewards_sparse.append(next_time_step.observation['state'][-2])

            ######### latents
            curr_action_taken, curr_latent_sample, curr_latent_dist = action_step.state
            # curr_latent_sample = (1, 1, [z1, z2])
            # curr_latent_dist = (1, 1, [z1_mean, z1_std, z2_mean, z2_std])
            z1_mean = curr_latent_dist[0, 0, :latent1_size]
            z1_std = curr_latent_dist[0, 0, latent1_size:2*latent1_size]
            z2_mean = curr_latent_dist[0, 0, 2*latent1_size:-latent2_size]
            z2_std = curr_latent_dist[0, 0, -latent2_size:]

            z1_means.append(np.linalg.norm(z1_mean))
            z1_stds.append(np.linalg.norm(z1_std))
            z2_means.append(np.linalg.norm(z2_mean))
            z2_stds.append(np.linalg.norm(z2_std))

            ########## recon

            # sample latents + reconstruct rewards from the samples

            num_latent_samples = 30

            cov = np.zeros((z1_mean.shape[0], z1_mean.shape[0]))
            np.fill_diagonal(cov, z1_std[None])
            latent1_samples = np.random.multivariate_normal(z1_mean, cov, num_latent_samples) #(num_latent_samples, z1_dim)
            latent1_samples = latent1_samples.astype(np.float32)

            cov2 = np.zeros((z2_mean.shape[0], z2_mean.shape[0]))
            np.fill_diagonal(cov2, z2_std[None])
            latent2_samples = np.random.multivariate_normal(z2_mean, cov2, num_latent_samples) #(num_latent_samples, z2_dim)
            latent2_samples = latent2_samples.astype(np.float32)

            #(num_latent_samples, 1, z1_dim), (num_latent_samples, 1, z2_dim)
            # rew_dists = policy._tf_policy._model_network.reward_decoder(latent1_samples[:,None,:], latent2_samples[:,None,:])
            # rew_means, rew_stds = session.run([rew_dists.mean(), rew_dists.stddev()]) #(num_samples, 1), (num_samples, 1)

            rew_means, rew_stds = session.run(decode_rews_op, feed_dict={latent_samples_1_ph: latent1_samples[:,None,:], 
                                                                          latent_samples_2_ph: latent2_samples[:,None,:],
                                                                        })

            # plot
            mean_of_means.append(np.mean(rew_means))
            std_of_means.append(np.std(rew_means))
            mean_of_vars.append(np.mean(rew_stds))
            std_of_vars.append(np.std(rew_stds))

        # run metrics
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        relevant_observers = eval_metrics[num_metrics_per_episode * episode:num_metrics_per_episode * (episode + 1)]
        for observer in relevant_observers:
          observer(traj)

        # save images for rendering
        if rendering:
          if 'pixels' in time_step.observation:
            # print("A STEP... ADDED...")
            images[-1].append(next_time_step.observation['pixels'])

        # update
        time_step = next_time_step
        policy_state = action_step.state

      # at end of episode, create plot of latents and rewards
      if not dont_plot_reward_gifs:

        dpi = 50
        width = 15
        height = 10
        fig = Figure(figsize=(width, height), dpi=dpi)
        # canvas = FigureCanvasAgg(fig)

        ######## latents
        ax = fig.add_subplot(121)

        # ax.plot(rewards_dense[1:], 'g', label='dense rew')
        ax.plot(rewards_sparse[1:], 'g--', label='sparse rew')

        ax.plot(z1_means, 'm', label='z1')
        ax.fill_between(np.arange(len(z1_means)), np.array(z1_means)+np.array(z1_stds), np.array(z1_means)-np.array(z1_stds), color='m', alpha=0.25)

        ax.plot(z2_means, 'c', label='z2')
        ax.fill_between(np.arange(len(z2_means)), np.array(z2_means)+np.array(z2_stds), np.array(z2_means)-np.array(z2_stds), color='c', alpha=0.25)

        ax.legend(loc='lower left')
        
        ######### reconstruction
        ax = fig.add_subplot(122)

        ax.plot(mean_of_means, 'r', label='recon reward mean')
        ax.fill_between(np.arange(len(mean_of_means)), np.array(mean_of_means)+np.array(std_of_means), np.array(mean_of_means)-np.array(std_of_means), color='r', alpha=0.25)

        ax.plot(mean_of_vars, 'b', label='recon reward var')
        ax.fill_between(np.arange(len(mean_of_means)), np.array(mean_of_vars)+np.array(std_of_vars), np.array(mean_of_vars)-np.array(std_of_vars), color='b', alpha=0.25)

        ax.plot(rewards_dense[1:], 'g', label='true dense rew')
        ax.plot(rewards_sparse[1:], 'g--', label='true sparse rew')

        ######### plot
        ax.legend(loc='lower left')
        # canvas.draw()
        # plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(dpi*width, dpi*height, 3)
        # variance_plots.append(plot_image)

        fig_list.append(fig)

      # render at the end of episode
      if rendering: 
        render_images.append(list(env.frames))
        env.frames[:] = []
        # print("ADDED NEW LIST AT END OF EPISODE...")
        images.append([])

      # one additional between-episode step to force the reset 
      # (ie transition between prev episode and next one, so step types all work out)

      # currently, time_step has step_type=2
      action_step = policy.action(mask_episode_transition(time_step), policy_state) # mask time_step
      # next_time_step is initial pos, action is ignored
      # step_type becomes 0
      next_time_step = env.step(action_step.action) 
      traj = trajectory.from_transition(time_step, action_step, next_time_step)

      # run metrics
      relevant_observers = eval_metrics[num_metrics_per_episode * episode:num_metrics_per_episode * (episode + 1)]
      for observer in relevant_observers:
        observer(traj) # observer might need these "boundary" transitions to properly separate episodes
      
      # render
      if rendering:
        if 'pixels' in time_step.observation:
          # print("BEGINNING OF EPISODE... ADDED...")
          images[-1].append(next_time_step.observation['pixels']) # first frame in next episode

      # update
      time_step = next_time_step # step_type=0
      policy_state = action_step.state

    num_finished_trials = trial + 1

    # when all episodes in a trial are done
    if rendering:
      # print("POPPED... len old:", len(images), " len new: ", len(images)-1)
      images.pop() # pop the empty list in the end

    # when all episodes in a trial are done AND not rendering anymore
    if rendering and num_finished_trials >= num_trials_to_render:
      rendering = False
      env.stop_rendering()
      
    # get ready for the next trial
    if rendering:
      # print("ADDED NEW LIST AT END OF A TRIAL...")
      images.append([])

  # when all trials are done
  if rendering:
    # print("POPPED when all trials are done... len old:", len(images)-1, " len new: ", len(images))
    images.pop()

  py_metric.run_summaries(eval_metrics)

  if eval_doing_second_agent:
    render_name='Eval2'
    obs_name='EvalObs2'
    rews_name='LatsRews2'
  else:
    render_name='Eval'
    obs_name='EvalObs'
    rews_name='LatsRews'

  # rendering images (what we want to see)
  if render_images:
    if not log_image_strips:
      render_images = np.array(render_images)
      render_images = np.transpose(render_images, [0,1,4,2,3])
      logger.log_video(render_images, render_name, step=global_step_val, fps=render_fps) #"Need [N, T, C, H, W] input tensor for video logging!"
    else:
      # images: num rollouts, rollout len, height, width, 3
      # [B,T,H,W,C] --> [B,H,T,W,C] --> [BH,T,W,C] --> [BH,TW,C]
      # This tiles batches vertically and time horizontally
      render_images = np.array(render_images)
      render_images = render_images[:,::5] # subsample
      render_images_tiled = np.transpose(render_images, [0,2,1,3,4])
      render_images_tiled = render_images_tiled.reshape((render_images_tiled.shape[0]*render_images_tiled.shape[1], render_images_tiled.shape[2], render_images_tiled.shape[3], render_images_tiled.shape[4]))
      render_images_tiled = render_images_tiled.reshape((render_images_tiled.shape[0], render_images_tiled.shape[1]*render_images_tiled.shape[2], render_images_tiled.shape[3])) #[BH,TW,C]
      render_images_tiled = np.transpose(render_images_tiled, (2, 0, 1)) #[C,BH,TW]
      logger.log_image(render_images_tiled, render_name, step=global_step_val) # [C, H, W]

  # observation images (what the robot sees)
  if log_image_observations:
    if images and not (model_input == 'state'):
      if not log_image_strips:
        images = np.array(images)
        images = np.transpose(images, [0,1,4,2,3])
        logger.log_video(images, obs_name, step=global_step_val, fps=render_fps) #"Need [N, T, C, H, W] input tensor for video logging!"
      else:
        # images: num rollouts, rollout len, height, width, 3
        # [B,T,H,W,C] --> [B,H,T,W,C] --> [BH,T,W,C] --> [BH,TW,C]
        # This tiles batches vertically and time horizontally
        images = np.array(images)
        images = images[:,::5] # subsample
        images_tiled = np.transpose(images, [0,2,1,3,4])
        images_tiled = images_tiled.reshape((images_tiled.shape[0]*images_tiled.shape[1], images_tiled.shape[2], images_tiled.shape[3], images_tiled.shape[4]))
        images_tiled = images_tiled.reshape((images_tiled.shape[0], images_tiled.shape[1]*images_tiled.shape[2], images_tiled.shape[3])) #[BH,TW,C]
        images_tiled = np.transpose(images_tiled, (2, 0, 1)) #[C,BH,TW]
        logger.log_image(images_tiled, obs_name, step=global_step_val) # [C, H, W]

  if not dont_plot_reward_gifs:
    for fig in fig_list:
      logger.log_figure(fig, rews_name, step=global_step_val, phase=0)

  logger.flush()

  if not dont_plot_reward_gifs:
    # these 3 things are for freeing things from mem
    fig.clf()
    plt.close()
  num = gc.collect()

  return


#######################################################
#######################################################


def perform_eval_and_summaries(metrics,
                      env,
                      policy,
                      num_episodes=1,
                      num_episodes_to_render=1,
                      images_ph=None,
                      images_summary=None,
                      render_images_summary=None,
                      eval_tasks=None,
                      model_input='image'):
  """
  Collect num_episodes on env, using policy
  Render num_episodes_to_render of those episodes
  eval_tasks: task list from which to reset the task for each episode
  Run summaries on the images

  NOTE: this function is an episode version (not trials)
  So the latent variable is reset after each episode.
  And task is reset for each episode.
  """

  ###########
  # init
  ###########

  for metric in metrics:
    metric.reset()
  if num_episodes_to_render:
    env.start_rendering()

  #####################
  # reset env + task
  #####################

  if eval_tasks is not None:
    # can repeat, since might be evaluating more episodes than # of eval tasks
    which_tasks = np.random.choice(len(eval_tasks), num_episodes)
    env.set_task_for_env(eval_tasks[which_tasks[0]])
  time_step = env.reset()

  #####################
  # init images holder
  #####################

  render_images = []
  if num_episodes_to_render and 'pixels' in time_step.observation:
    images = [[time_step.observation['pixels']]]
  else:
    images = []

  #################################
  # collect num_episodes rollouts
  #################################

  step = 0
  episode = 0
  policy_state_initial = policy.get_initial_state(env.batch_size)
  policy_state = policy.get_initial_state(env.batch_size)

  while episode < num_episodes:

    # at beginning of each episode
    if step > 0 and time_step.is_first():

      # reset task (to a random one, from the list)
      # Note: policy_state (i.e. latent z) gets automatically reset
      if eval_tasks is not None:
        task_idx = np.minimum(episode, len(which_tasks) - 1)
        task = which_tasks[task_idx]
        env.set_task_for_env(eval_tasks[task])

    # get action
    action_step = policy.action(time_step, policy_state)

    # take step
    next_time_step = env.step(action_step.action)

    # run metrics on the step
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    for observer in metrics:
      observer(traj)

    # render
    if episode < num_episodes_to_render:
      if traj.is_last():
        render_images.append(list(env.frames))
        env.frames[:] = []
        if episode + 1 >= num_episodes_to_render:
          env.stop_rendering()

      if 'pixels' in time_step.observation:
        if traj.is_boundary():
          images.append([])
        images[-1].append(next_time_step.observation['pixels'])

    # +1 episode if transition is end of a rollout
    if traj.is_last():
      # add to episode counter
      episode += 1

    # +1 step if it's not a boundary transition
    step += np.sum(~traj.is_boundary())

    time_step = next_time_step
    policy_state = action_step.state

  py_metric.run_summaries(metrics)

  if render_images:
    render_images = pad_and_concatenate_videos(render_images)
    session = tf.compat.v1.get_default_session()
    session.run(render_images_summary, feed_dict={images_ph: [render_images]})
  if images and not (model_input == 'state'):
    images = pad_and_concatenate_videos(images)
    session = tf.compat.v1.get_default_session()
    session.run(images_summary, feed_dict={images_ph: [images]})


