from tf_agents.utils import common
from meld.utils.logger import Logger
from meld.utils.utils import *
from meld.utils.eval_summaries import perform_eval_and_summaries_meld

import IPython
e = IPython.embed

def load_eval_log(train_eval_dir,
                  meld_agent,
                  global_step,
                  sess,
                  eval_metrics,
                  eval_py_env,
                  eval_py_policy,
                  num_eval_trials,
                  max_episode_len,
                  episodes_per_trial,
                  log_image_strips,
                  num_trials_to_render,
                  train_tasks, # in case want to eval on a train task
                  eval_tasks,
                  model_net,
                  render_fps,
                  decode_rews_op,
                  latent_samples_1_ph,
                  latent_samples_2_ph,

                  eval_second_pol=False,

                  **kwargs
                  ):

  print("\n\nSTARTING OFFLINE EVAL!")
  print("Will do eval for ", num_eval_trials, " trials, with ", episodes_per_trial, " episides per trial.")
  print("Will render ", num_trials_to_render, "/", num_eval_trials, " trials.")

  # Where to save the new gifs from this offline eval
  eval_dir_after = os.path.join(train_eval_dir, 'eval_offline')
  eval_logger_after = Logger(eval_dir_after)

  eval_tasks_orig = eval_tasks

  ######################
  print("\n\nBREAKPOINT... Can do manual eval...\n\n")
  e()

  # temp overwrite these vars for offline eval
  num_eval_trials = 4
  num_trials_to_render = 4
  eval_tasks = [eval_tasks[0]] # single task index, will force all trials below to be for same task

  ######################

  eval_global_steps_list = [2000000] # Input the checkpoint global step you want to eval here
  for eval_global_step in eval_global_steps_list:

    actual_loaded_step, policy_status = load_pol_ckpt(train_eval_dir, sess, eval_global_step, meld_agent, global_step, eval_second_pol)
    print(f"loaded step {actual_loaded_step}\npolicy status: {policy_status}")


    for m in eval_metrics:
      m.reset()

    # Eval for num_eval_trials + log gifs to tb 
    # The slider of GIF will be reflecting the true global step that policy is loading from
    for eval_iter in range(1):
      perform_eval_and_summaries_meld(
        eval_metrics,
        eval_py_env,
        eval_py_policy,
        num_eval_trials,
        max_episode_len,
        episodes_per_trial,
        log_image_strips=log_image_strips,
        num_trials_to_render_overwrite=num_trials_to_render, # overwrite the default "1" to visualize more
        eval_tasks=eval_tasks,
        latent1_size=model_net.latent1_size,
        latent2_size=model_net.latent2_size,
        logger=eval_logger_after,
        global_step_val=sess.run(global_step),
        render_fps=render_fps,
        decode_rews_op=decode_rews_op,
        latent_samples_1_ph=latent_samples_1_ph,
        latent_samples_2_ph=latent_samples_2_ph,
        dont_plot_reward_gifs=False, ## plot the reward predics
        log_image_observations=True, ## log image obs that robot sees
      )

    #print out the avg of reward and scores across num_eval_trials
    for m in eval_metrics:
      print(m.name[2:], m.result()) 



def load_pol_ckpt(train_eval_dir, sess, eval_global_step, meld_agent, global_step, eval_second_pol):
  train_dir = os.path.join(train_eval_dir, 'train')

  pol_name = 'policy'
  if eval_second_pol:
    pol_name = 'policy2'

  actual_loaded_step = set_loading_step(train_dir, eval_global_step, pol_name)

  policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir, pol_name),
    policy=meld_agent.policy,
    global_step=global_step,
    max_to_keep=99999999999)  # keep many policy checkpoints, in case of future eval

  policy_status = policy_checkpointer.initialize_or_restore(sess)

  # Initialize variables
  common.initialize_uninitialized_variables(sess)

  set_global_step(global_step, sess, actual_loaded_step)

  # make the checkpoint file pointing back to the latest checkpoint
  set_loading_step(train_dir, step=None)

  return actual_loaded_step, policy_status



def set_loading_step(ckpt_dir, step=None, pol_name='policy'):
  """
  The checkpointer does NOT come with an option to specify which checkpoint to load. Instead it will always load
  the file specified in policy/checkpoint. During normal training, it will be pointing to the latest file. Here
  we change policy/checkpoint such that it is pointing to the timestep we want to load.
  """
  # read the file and find all checkpoints
  with open(os.path.join(ckpt_dir, pol_name+"/checkpoint"), "r") as f:
    log = f.readlines()
  all_ckpts = []
  for line in log:
    if "all_model_checkpoint_paths:" in line:
      num = int(line[34:-2])
      all_ckpts.append(num)
  all_ckpts.sort()

  # find the nearest one to the input step
  load_step = None
  if not step is None:
    for s in all_ckpts:
      if s > step:
        load_step = s
        break
    if load_step is None:
      load_step = all_ckpts[-1]
  else:
    load_step = all_ckpts[-1]

  # modify the first line of that file
  first_line = log[0]
  new_first_line = first_line[:29] + str(load_step) + first_line[-2:]
  log[0] = new_first_line
  with open(os.path.join(ckpt_dir, pol_name+"/checkpoint"), "w") as f:
    f.write("".join(log))
  with open(os.path.join(ckpt_dir, pol_name+"/checkpoint"), "r") as f:
    log = f.readlines()
    print("\nCheckpoint to load:\n", log[0], "\n")

  return load_step




def set_global_step(global_step, sess, num_steps):
  # print("before: ", _sess.run(_global_step))
  op = global_step.assign(num_steps)
  sess.run(op)
  after_val = sess.run(global_step)
  return after_val

