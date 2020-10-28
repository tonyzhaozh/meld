from gym.envs.registration import register


###################################

def register_all_gym_envs(MAX_STEPS_DICT):

    print("Registering gym environments")

    register(
        id='HalfCheetahVel-v0',
        entry_point='meld.environments.envs.cheetah.cheetah_vel:HalfCheetahVelEnv',
        max_episode_steps=MAX_STEPS_DICT["HalfCheetahVel-v0"],
    )

    ###################################

    register(
        id='SawyerReach-v0',
        entry_point='meld.environments.envs.reacher.sawyer_reacher:SawyerReachingEnv',
        max_episode_steps=MAX_STEPS_DICT["SawyerReach-v0"],
    )

    register(
        id='SawyerReachMT-v0',
        entry_point='meld.environments.envs.reacher.sawyer_reacher:SawyerReachingEnvMultitask',
        max_episode_steps=MAX_STEPS_DICT["SawyerReachMT-v0"],
    )

    ###################################

    register(
        id='SawyerPeg-v0',
        entry_point='meld.environments.envs.peg.sawyer_peg:SawyerPegInsertionEnv',
        max_episode_steps=MAX_STEPS_DICT["SawyerPeg-v0"],
    )

    register(
        id='SawyerPegMT-v0',
        entry_point='meld.environments.envs.peg.sawyer_peg:SawyerPegInsertionEnvMultitask',
        max_episode_steps=MAX_STEPS_DICT["SawyerPegMT-v0"],
    )

    register(
        id='SawyerPegMT4box-v0',
        entry_point='meld.environments.envs.peg.sawyer_peg:SawyerPegInsertionEnv4Box',
        max_episode_steps=MAX_STEPS_DICT["SawyerPegMT4box-v0"],
    )

    ###################################

    register(
        id='SawyerButtons-v0',
        entry_point='meld.environments.envs.button.sawyer_button:SawyerButtonsEnv',
        max_episode_steps=MAX_STEPS_DICT["SawyerButtons-v0"],
    )


    ###################################

    register(
        id='SawyerShelfMT-v0',
        entry_point='meld.environments.envs.shelf.sawyer_shelf:SawyerPegShelfEnvMultitask',
        max_episode_steps=MAX_STEPS_DICT["SawyerShelfMT-v0"],
    )

