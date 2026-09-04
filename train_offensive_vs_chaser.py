# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:27:52 2026

@author: nibra
"""

# -*- coding: utf-8 -*-

"""
OFFENSIVE AIR-HOCKEY - STAGE 2
==============================

BLUE:
    Previously-trained V5 offensive PPO agent.

RED:
    Defensive puck-chasing bot.

    - Moves at 50% of BLUE'S maximum paddle speed.
    - If puck is in RED'S half:
          chase the puck.
    - If puck is in BLUE'S half:
          return to the centre of RED'S half and wait.


TRAINING
========

5 Optuna trials.

Each trial continues 3 previously-trained agents:

    V5 Trial 3 Seed 1
    V5 Trial 3 Seed 2
    V5 Trial 3 Seed 3

Each receives:

    +200,000 timesteps

against the half-speed defensive chaser.


OPTUNA OBJECTIVE
================

Optuna maximises the MEDIAN blue goals/game
across the 3 agents.

Example:

    Agent scores:
        27
        0
        0

    Mean   = 9
    Median = 0

Therefore one lucky agent cannot make a reward
system look good if the other two fail.


REWARDS
=======

Goal:
    +1.0 fixed

Hit puck:
    tuned by Optuna

Chase start:
    +0.03 fixed

Useful chase only if:

    - puck is in BLUE'S half
    - blue is behind puck
    - blue changes into useful velocity
    - velocity is within 45 degrees of puck direction
    - chase cooldown has expired

Opponent touch:
    tuned by Optuna

Opponent block:
    tuned by Optuna

Chase cooldown:
    tuned by Optuna
"""


# ================================================================
# IMPORTS
# ================================================================

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from air_hockey_env import AirHockeyEnv


# ================================================================
# QUICK TEST / FULL RUN
# ================================================================

QUICK_TEST = False


if QUICK_TEST:

    TARGET_TRIALS = 3

    CONTINUED_TIMESTEPS = 10_000

    EVALUATION_GAMES = 5

    EVALUATION_SECONDS = 10


else:

    TARGET_TRIALS = 5

    CONTINUED_TIMESTEPS = 200_000

    EVALUATION_GAMES = 100

    EVALUATION_SECONDS = 60


AGENTS_PER_TRIAL = 3


# ================================================================
# SIMULATION
# ================================================================

SIMULATION_FPS = 60


TRAIN_MATCH_SECONDS = 60


TRAIN_EPISODE_MAX_STEPS = (
    TRAIN_MATCH_SECONDS
    *
    SIMULATION_FPS
)


EVALUATION_STEPS_PER_GAME = (
    EVALUATION_SECONDS
    *
    SIMULATION_FPS
)


# ================================================================
# RED CHASER SPEED
# ================================================================

# Red moves at 50% of blue's maximum paddle speed.

CHASER_SPEED_MULTIPLIER = 0.50


# ================================================================
# STARTING V5 MODELS
# ================================================================

V5_FOLDER = (
    "offensive_reward_tuning_v5"
)


V5_STARTING_TRIAL = 3


TRAINING_SEEDS = [
    1,
    2,
    3
]


def get_starting_model_path(
    seed
):

    return os.path.join(

        V5_FOLDER,

        "models",

        (
            f"trial_{V5_STARTING_TRIAL}"
            f"_seed_{seed}.zip"
        )
    )


# ================================================================
# EVALUATION SEEDS
# ================================================================

EVALUATION_SEED_START = 30_000


# ================================================================
# FIXED REWARDS
# ================================================================

GOAL_REWARD = 1.0


CHASE_START_REWARD = 0.03


CHASE_MAX_ANGLE_DEGREES = 45


# ================================================================
# V5 TRIAL 3 BASELINE
# ================================================================

# These were the winning V5 values.

BASELINE_HIT_REWARD = (
    0.14193125727042166
)


BASELINE_TOUCH_PENALTY = (
    -0.30334147276556134
)


BASELINE_BLOCK_PENALTY = (
    -0.053228220396567494
)


BASELINE_CHASE_COOLDOWN = 41


# ================================================================
# OPTUNA SEARCH RANGES
# ================================================================

HIT_REWARD_MIN = 0.0

HIT_REWARD_MAX = 0.25


TOUCH_PENALTY_MIN = -0.75

TOUCH_PENALTY_MAX = 0.0


BLOCK_PENALTY_MIN = -1.50

BLOCK_PENALTY_MAX = 0.0


CHASE_COOLDOWN_MIN = 5

CHASE_COOLDOWN_MAX = 60


# ================================================================
# OUTPUT
# ================================================================

# NEW folder/database.
#
# This does NOT mix with the previous full-speed chaser run.

if QUICK_TEST:

    OUTPUT_FOLDER = (
        "offensive_vs_half_chaser_stage2_test"
    )

else:

    OUTPUT_FOLDER = (
        "offensive_vs_half_chaser_stage2"
    )


MODEL_FOLDER = os.path.join(

    OUTPUT_FOLDER,

    "models"
)


os.makedirs(

    MODEL_FOLDER,

    exist_ok=True
)


DATABASE_PATH = os.path.join(

    OUTPUT_FOLDER,

    "optuna_study.db"
)


CSV_PATH = os.path.join(

    OUTPUT_FOLDER,

    "all_trials.csv"
)


STUDY_NAME = (
    "offensive_vs_half_chaser_stage2"
)


# ================================================================
# ENVIRONMENT
# ================================================================

class HalfSpeedDefensiveChaserEnv(
    AirHockeyEnv
):

    def __init__(
        self,
        hit_reward,
        opponent_touch_penalty,
        block_penalty,
        chase_cooldown_steps,
        render_mode=None,
        training_mode=True
    ):

        super().__init__(
            render_mode=render_mode
        )


        # ============================================================
        # REWARDS
        # ============================================================

        self.goal_reward = (
            GOAL_REWARD
        )


        self.hit_reward = float(
            hit_reward
        )


        self.opponent_touch_penalty = float(
            opponent_touch_penalty
        )


        self.block_penalty = float(
            block_penalty
        )


        self.chase_cooldown_steps = int(
            chase_cooldown_steps
        )


        # ============================================================
        # MODE
        # ============================================================

        self.training_mode = (
            training_mode
        )


        # ============================================================
        # PPO CONTROLS BLUE ONLY
        # ============================================================

        self.action_space = spaces.Box(

            low=-1.0,

            high=1.0,

            shape=(2,),

            dtype=np.float32
        )


        # ============================================================
        # MATCH STATE
        # ============================================================

        self.episode_steps = 0


        self.ai_shot_active = False


        self.chase_reward_cooldown = 0


    # ================================================================
    # NORMAL RESET
    # ================================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        observation, info = super().reset(

            seed=seed,

            options=options
        )


        self.episode_steps = 0


        self.ai_shot_active = False


        self.chase_reward_cooldown = 0


        return observation, info


    # ================================================================
    # RESET AFTER GOAL
    # ================================================================

    def reset_after_goal(
        self,
        conceding_player
    ):

        # ============================================================
        # BLUE PADDLE
        # ============================================================

        self.paddle1_pos = np.array(

            [
                self.width / 2,
                self.height * 0.75
            ],

            dtype=np.float32
        )


        self.paddle1_vel = np.zeros(

            2,

            dtype=np.float32
        )


        # ============================================================
        # RED PADDLE
        # ============================================================

        self.paddle2_pos = np.array(

            [
                self.width / 2,
                self.height * 0.25
            ],

            dtype=np.float32
        )


        self.paddle2_vel = np.zeros(

            2,

            dtype=np.float32
        )


        # ============================================================
        # PUCK RESTART
        # ============================================================

        centre_y = (
            self.height / 2
        )


        one_third_of_half = (

            (self.height / 2)

            /

            3
        )


        # ------------------------------------------------------------
        # BLUE CONCEDED
        # ------------------------------------------------------------

        if conceding_player == 1:

            puck_y = (

                centre_y

                +

                one_third_of_half
            )


        # ------------------------------------------------------------
        # RED CONCEDED
        # ------------------------------------------------------------

        elif conceding_player == 2:

            puck_y = (

                centre_y

                -

                one_third_of_half
            )


        else:

            raise ValueError(
                "conceding_player must be 1 or 2"
            )


        self.puck_pos = np.array(

            [
                self.width / 2,
                puck_y
            ],

            dtype=np.float32
        )


        # Puck starts stationary.

        self.puck_vel = np.zeros(

            2,

            dtype=np.float32
        )


        # New rally.

        self.ai_shot_active = False


        self.chase_reward_cooldown = 0


        return self._get_observation()


    # ================================================================
    # RED DEFENSIVE BEHAVIOUR
    # ================================================================

    def _red_defensive_action(
        self
    ):

        """
        RED:

        If puck is in RED'S half:
            chase puck.

        If puck is in BLUE'S half:
            return to centre of RED'S half.

        Red never teleports.
        It moves normally toward whichever target applies.
        """


        # ============================================================
        # PUCK IN RED HALF?
        # ============================================================

        puck_in_red_half = (

            self.puck_pos[1]

            <

            self.height / 2
        )


        # ============================================================
        # TARGET = PUCK
        # ============================================================

        if puck_in_red_half:

            target_position = (
                self.puck_pos.copy()
            )


        # ============================================================
        # TARGET = WAITING POSITION
        # ============================================================

        else:

            target_position = np.array(

                [
                    self.width / 2,
                    self.height * 0.25
                ],

                dtype=np.float32
            )


        # ============================================================
        # VECTOR RED -> TARGET
        # ============================================================

        direction = (

            target_position

            -

            self.paddle2_pos
        )


        distance = float(

            np.linalg.norm(
                direction
            )
        )


        # ============================================================
        # ALREADY AT TARGET
        # ============================================================

        if distance <= 1e-8:

            return np.zeros(

                2,

                dtype=np.float32
            )


        # ============================================================
        # NORMALISE
        # ============================================================

        action = (

            direction

            /

            distance
        )


        return action.astype(
            np.float32
        )


    # ================================================================
    # DISTANCE TO RED GOAL
    # ================================================================

    def _distance_to_opponent_goal(
        self,
        position
    ):

        goal_centre = np.array(

            [
                self.width / 2,
                0.0
            ],

            dtype=np.float32
        )


        return float(

            np.linalg.norm(

                position

                -

                goal_centre
            )
        )


    # ================================================================
    # PUCK HEADING TOWARD RED GOAL?
    # ================================================================

    def _puck_heading_towards_goal(
        self
    ):

        current_distance = (
            self._distance_to_opponent_goal(
                self.puck_pos
            )
        )


        predicted_position = (

            self.puck_pos

            +

            self.puck_vel
        )


        predicted_distance = (
            self._distance_to_opponent_goal(
                predicted_position
            )
        )


        return (

            predicted_distance

            <

            current_distance
        )


    # ================================================================
    # USEFUL BLUE CHASE?
    # ================================================================

    def _is_useful_chase_velocity(
        self,
        velocity
    ):

        # ============================================================
        # PUCK MUST BE IN BLUE HALF
        # ============================================================

        puck_in_blue_half = (

            self.puck_pos[1]

            >

            self.height / 2
        )


        if not puck_in_blue_half:

            return False


        # ============================================================
        # BLUE MUST BE BEHIND PUCK
        # ============================================================

        blue_is_behind_puck = (

            self.paddle1_pos[1]

            >

            self.puck_pos[1]
        )


        if not blue_is_behind_puck:

            return False


        # ============================================================
        # VECTOR BLUE -> PUCK
        # ============================================================

        to_puck = (

            self.puck_pos

            -

            self.paddle1_pos
        )


        puck_distance = float(

            np.linalg.norm(
                to_puck
            )
        )


        if puck_distance <= 1e-8:

            return False


        # ============================================================
        # BLUE SPEED
        # ============================================================

        speed = float(

            np.linalg.norm(
                velocity
            )
        )


        if speed <= 1e-8:

            return False


        # ============================================================
        # ANGLE
        # ============================================================

        cosine_angle = float(

            np.dot(

                velocity,

                to_puck
            )

            /

            (
                speed

                *

                puck_distance
            )
        )


        cosine_angle = float(

            np.clip(

                cosine_angle,

                -1.0,

                1.0
            )
        )


        minimum_cosine = float(

            np.cos(

                np.deg2rad(
                    CHASE_MAX_ANGLE_DEGREES
                )
            )
        )


        return (

            cosine_angle

            >=

            minimum_cosine
        )


    # ================================================================
    # COLLISION + REAL HIT DETECTION
    # ================================================================

    def _collision_and_detect(
        self,
        paddle_pos,
        paddle_vel
    ):

        difference = (

            self.puck_pos

            -

            paddle_pos
        )


        distance = float(

            np.linalg.norm(
                difference
            )
        )


        minimum_distance = (

            self.puck_radius

            +

            self.paddle_radius
        )


        if distance >= minimum_distance:

            return False


        # ============================================================
        # COLLISION NORMAL
        # ============================================================

        if distance == 0:

            normal = np.array(

                [
                    1.0,
                    0.0
                ],

                dtype=np.float32
            )


        else:

            normal = (

                difference

                /

                distance
            )


        # ============================================================
        # REAL HIT CHECK
        # ============================================================

        puck_normal_speed = float(

            np.dot(

                self.puck_vel,

                normal
            )
        )


        paddle_normal_speed = float(

            np.dot(

                paddle_vel,

                normal
            )
        )


        relative_speed = (

            puck_normal_speed

            -

            paddle_normal_speed
        )


        real_hit = (

            relative_speed

            <

            0
        )


        # Original collision physics.

        super()._handle_paddle_collision(

            paddle_pos,

            paddle_vel
        )


        return real_hit


    # ================================================================
    # STEP
    # ================================================================

    def step(
        self,
        blue_action
    ):

        self.episode_steps += 1


        # ============================================================
        # BLUE ACTION
        # ============================================================

        blue_action = np.asarray(

            blue_action,

            dtype=np.float32
        )


        blue_action = np.clip(

            blue_action,

            -1.0,

            1.0
        )


        old_blue_velocity = (
            self.paddle1_vel.copy()
        )


        new_blue_velocity = (

            blue_action

            *

            self.paddle_speed
        )


        # ============================================================
        # RED DEFENSIVE ACTION
        # ============================================================

        red_action = (
            self._red_defensive_action()
        )


        # ============================================================
        # RED MOVES AT 50% SPEED
        # ============================================================

        new_red_velocity = (

            red_action

            *

            self.paddle_speed

            *

            CHASER_SPEED_MULTIPLIER
        )


        # ============================================================
        # REWARD
        # ============================================================

        reward = 0.0


        # ============================================================
        # CHASE COOLDOWN
        # ============================================================

        if self.chase_reward_cooldown > 0:

            self.chase_reward_cooldown -= 1


        old_velocity_useful = (
            self._is_useful_chase_velocity(
                old_blue_velocity
            )
        )


        new_velocity_useful = (
            self._is_useful_chase_velocity(
                new_blue_velocity
            )
        )


        velocity_change = float(

            np.linalg.norm(

                new_blue_velocity

                -

                old_blue_velocity
            )
        )


        velocity_changed = (

            velocity_change

            >

            1e-6
        )


        started_useful_chase = (

            velocity_changed

            and

            new_velocity_useful

            and

            not old_velocity_useful

            and

            self.chase_reward_cooldown == 0
        )


        if started_useful_chase:

            reward += (
                CHASE_START_REWARD
            )


            self.chase_reward_cooldown = (
                self.chase_cooldown_steps
            )


        # ============================================================
        # APPLY VELOCITIES
        # ============================================================

        self.paddle1_vel = (
            new_blue_velocity
        )


        self.paddle2_vel = (
            new_red_velocity
        )


        # ============================================================
        # MOVE PADDLES
        # ============================================================

        self.paddle1_pos += (
            self.paddle1_vel
        )


        self.paddle2_pos += (
            self.paddle2_vel
        )


        self._limit_paddles()


        # ============================================================
        # MOVE PUCK
        # ============================================================

        self.puck_pos += (
            self.puck_vel
        )


        self._handle_wall_collisions()


        # ============================================================
        # BLUE HIT
        # ============================================================

        player1_hit = (
            self._collision_and_detect(

                self.paddle1_pos,

                self.paddle1_vel
            )
        )


        if player1_hit:

            reward += (
                self.hit_reward
            )


            self.ai_shot_active = True


        # ============================================================
        # IS ACTIVE SHOT GOING TO RED GOAL?
        # ============================================================

        was_heading_towards_goal = (

            self.ai_shot_active

            and

            self._puck_heading_towards_goal()
        )


        # ============================================================
        # RED HIT
        # ============================================================

        player2_hit = (
            self._collision_and_detect(

                self.paddle2_pos,

                self.paddle2_vel
            )
        )


        if player2_hit:

            reward += (
                self.opponent_touch_penalty
            )


            if was_heading_towards_goal:

                reward += (
                    self.block_penalty
                )


            self.ai_shot_active = False


        # ============================================================
        # GOAL
        # ============================================================

        goal = (
            self._check_goal()
        )


        terminated = False

        truncated = False


        # ============================================================
        # BLUE SCORES
        # ============================================================

        if goal == 1:

            reward += (
                self.goal_reward
            )


            observation = (
                self.reset_after_goal(
                    conceding_player=2
                )
            )


        # ============================================================
        # RED SCORES
        # ============================================================

        elif goal == 2:

            # No direct concede penalty.

            observation = (
                self.reset_after_goal(
                    conceding_player=1
                )
            )


        # ============================================================
        # NO GOAL
        # ============================================================

        else:

            observation = (
                self._get_observation()
            )


        # ============================================================
        # END TRAINING MATCH
        # ============================================================

        if (

            self.training_mode

            and

            self.episode_steps
            >=
            TRAIN_EPISODE_MAX_STEPS

        ):

            truncated = True


        # ============================================================
        # INFO
        # ============================================================

        info = {

            "goal":
                goal,

            "player1_hit":
                player1_hit,

            "player2_hit":
                player2_hit,

            "started_useful_chase":
                started_useful_chase,

            "chase_cooldown_remaining":
                self.chase_reward_cooldown
        }


        if self.render_mode == "human":

            self.render()


        return (

            observation,

            float(reward),

            terminated,

            truncated,

            info
        )


# ================================================================
# MAKE ENV
# ================================================================

def make_env(
    hit_reward,
    opponent_touch_penalty,
    block_penalty,
    chase_cooldown_steps,
    training_mode=True,
    render_mode=None
):

    return HalfSpeedDefensiveChaserEnv(

        hit_reward=hit_reward,

        opponent_touch_penalty=(
            opponent_touch_penalty
        ),

        block_penalty=block_penalty,

        chase_cooldown_steps=(
            chase_cooldown_steps
        ),

        training_mode=training_mode,

        render_mode=render_mode
    )


# ================================================================
# EVALUATE ONE AGENT
# ================================================================

def evaluate_agent(
    model,
    hit_reward,
    opponent_touch_penalty,
    block_penalty,
    chase_cooldown_steps
):

    env = make_env(

        hit_reward,

        opponent_touch_penalty,

        block_penalty,

        chase_cooldown_steps,

        training_mode=False
    )


    blue_scores = []

    red_scores = []


    # ================================================================
    # EVALUATION GAMES
    # ================================================================

    for game_number in range(
        EVALUATION_GAMES
    ):

        observation, info = env.reset(

            seed=(

                EVALUATION_SEED_START

                +

                game_number
            )
        )


        blue_goals = 0

        red_goals = 0


        # ============================================================
        # PLAY GAME
        # ============================================================

        for _ in range(
            EVALUATION_STEPS_PER_GAME
        ):

            action, _ = model.predict(

                observation,

                deterministic=True
            )


            (
                observation,
                reward,
                terminated,
                truncated,
                info

            ) = env.step(
                action
            )


            if info["goal"] == 1:

                blue_goals += 1


            elif info["goal"] == 2:

                red_goals += 1


        blue_scores.append(
            blue_goals
        )


        red_scores.append(
            red_goals
        )


    env.close()


    blue_scores = np.asarray(

        blue_scores,

        dtype=np.float32
    )


    red_scores = np.asarray(

        red_scores,

        dtype=np.float32
    )


    # ================================================================
    # RESULTS
    # ================================================================

    return {

        "average_blue_goals":

            float(
                np.mean(
                    blue_scores
                )
            ),


        "median_blue_goals":

            float(
                np.median(
                    blue_scores
                )
            ),


        "average_red_goals":

            float(
                np.mean(
                    red_scores
                )
            ),


        "average_goal_difference":

            float(
                np.mean(

                    blue_scores

                    -

                    red_scores
                )
            )
    }


# ================================================================
# OPTUNA OBJECTIVE
# ================================================================

def objective(
    trial
):

    print()
    print("=" * 70)

    print(
        f"HALF-SPEED CHASER - "
        f"TRIAL {trial.number + 1}"
    )

    print("=" * 70)


    # ============================================================
    # PARAMETERS
    # ============================================================

    hit_reward = trial.suggest_float(

        "hit_reward",

        HIT_REWARD_MIN,

        HIT_REWARD_MAX
    )


    opponent_touch_penalty = (
        trial.suggest_float(

            "opponent_touch_penalty",

            TOUCH_PENALTY_MIN,

            TOUCH_PENALTY_MAX
        )
    )


    block_penalty = (
        trial.suggest_float(

            "block_penalty",

            BLOCK_PENALTY_MIN,

            BLOCK_PENALTY_MAX
        )
    )


    chase_cooldown_steps = (
        trial.suggest_int(

            "chase_cooldown_steps",

            CHASE_COOLDOWN_MIN,

            CHASE_COOLDOWN_MAX
        )
    )


    # ============================================================
    # PRINT REWARD SYSTEM
    # ============================================================

    print()
    print("Reward system:")


    print(
        f"Goal:              "
        f"{GOAL_REWARD:+.4f}"
    )


    print(
        f"Hit puck:          "
        f"{hit_reward:+.4f}"
    )


    print(
        f"Chase start:       "
        f"{CHASE_START_REWARD:+.4f}"
    )


    print(
        f"Chase max angle:   "
        f"{CHASE_MAX_ANGLE_DEGREES} degrees"
    )


    print(
        f"Chase cooldown:    "
        f"{chase_cooldown_steps} steps"
    )


    print(
        f"Opponent touch:    "
        f"{opponent_touch_penalty:+.4f}"
    )


    print(
        f"Opponent block:    "
        f"{block_penalty:+.4f}"
    )


    print(
        f"Red speed:         "
        f"{CHASER_SPEED_MULTIPLIER * 100:.0f}%"
    )


    # ============================================================
    # AGENT RESULTS
    # ============================================================

    agent_scores = []

    agent_red_scores = []

    agent_goal_differences = []


    # ============================================================
    # CONTINUE THREE V5 AGENTS
    # ============================================================

    for agent_number, seed in enumerate(

        TRAINING_SEEDS,

        start=1
    ):

        print()
        print(
            f"--- Agent "
            f"{agent_number}/3 "
            f"| starting V5 seed {seed} ---"
        )


        starting_model_path = (
            get_starting_model_path(
                seed
            )
        )


        if not os.path.exists(
            starting_model_path
        ):

            raise FileNotFoundError(

                "\nCould not find:\n"
                f"{starting_model_path}\n"
            )


        # ============================================================
        # TRAINING ENV
        # ============================================================

        env = make_env(

            hit_reward,

            opponent_touch_penalty,

            block_penalty,

            chase_cooldown_steps,

            training_mode=True
        )


        env.reset(
            seed=seed
        )


        # ============================================================
        # LOAD EXISTING PPO
        # ============================================================

        model = PPO.load(

            starting_model_path,

            env=env,

            device="auto"
        )


        model.set_random_seed(
            seed
        )


        # ============================================================
        # CONTINUE TRAINING
        # ============================================================

        print(

            f"Continuing training for "
            f"{CONTINUED_TIMESTEPS:,} "
            f"timesteps..."
        )


        model.learn(

            total_timesteps=(
                CONTINUED_TIMESTEPS
            ),

            reset_num_timesteps=False
        )


        # ============================================================
        # SAVE MODEL
        # ============================================================

        model_path = os.path.join(

            MODEL_FOLDER,

            (
                f"trial_{trial.number + 1}"
                f"_seed_{seed}"
            )
        )


        model.save(
            model_path
        )


        env.close()


        # ============================================================
        # EVALUATE
        # ============================================================

        print(
            f"Evaluating seed {seed}..."
        )


        evaluation = evaluate_agent(

            model,

            hit_reward,

            opponent_touch_penalty,

            block_penalty,

            chase_cooldown_steps
        )


        blue_score = (
            evaluation[
                "average_blue_goals"
            ]
        )


        red_score = (
            evaluation[
                "average_red_goals"
            ]
        )


        goal_difference = (
            evaluation[
                "average_goal_difference"
            ]
        )


        agent_scores.append(
            blue_score
        )


        agent_red_scores.append(
            red_score
        )


        agent_goal_differences.append(
            goal_difference
        )


        print(

            f"Seed {seed} -> "
            f"{blue_score:.3f} "
            f"blue goals/game"
        )


    # ============================================================
    # TRIAL STATISTICS
    # ============================================================

    average_blue_goals = float(

        np.mean(
            agent_scores
        )
    )


    median_blue_goals = float(

        np.median(
            agent_scores
        )
    )


    std_between_agents = float(

        np.std(
            agent_scores
        )
    )


    average_red_goals = float(

        np.mean(
            agent_red_scores
        )
    )


    average_goal_difference = float(

        np.mean(
            agent_goal_differences
        )
    )


    # ============================================================
    # PRINT RESULT
    # ============================================================

    print()
    print(
        "TRIAL RESULT"
    )


    for index, score in enumerate(

        agent_scores,

        start=1
    ):

        print(

            f"Agent {index}: "
            f"{score:.3f}"
        )


    print(
        f"Mean across 3 agents:   "
        f"{average_blue_goals:.3f}"
    )


    print(
        f"Median across 3 agents: "
        f"{median_blue_goals:.3f}"
    )


    print(
        f"Variation:              "
        f"{std_between_agents:.3f}"
    )


    print(
        f"Average red goals:      "
        f"{average_red_goals:.3f}"
    )


    print(
        f"Average goal difference:"
        f" {average_goal_difference:+.3f}"
    )


    # ============================================================
    # SAVE EXTRA DETAILS
    # ============================================================

    trial.set_user_attr(
        "agent_1_score",
        agent_scores[0]
    )


    trial.set_user_attr(
        "agent_2_score",
        agent_scores[1]
    )


    trial.set_user_attr(
        "agent_3_score",
        agent_scores[2]
    )


    trial.set_user_attr(
        "average_blue_goals",
        average_blue_goals
    )


    trial.set_user_attr(
        "std_between_agents",
        std_between_agents
    )


    trial.set_user_attr(
        "average_red_goals",
        average_red_goals
    )


    trial.set_user_attr(
        "average_goal_difference",
        average_goal_difference
    )


    # ============================================================
    # OPTUNA MAXIMISES MEDIAN
    # ============================================================

    return median_blue_goals


# ================================================================
# SAVE CSV
# ================================================================

def save_results_csv(
    study
):

    rows = []


    for trial in study.trials:

        if (
            trial.state
            !=
            optuna.trial.TrialState.COMPLETE
        ):

            continue


        rows.append({

            "trial":
                trial.number + 1,


            "median_blue_goals":
                trial.value,


            "average_blue_goals":
                trial.user_attrs.get(
                    "average_blue_goals",
                    np.nan
                ),


            "hit_reward":
                trial.params[
                    "hit_reward"
                ],


            "opponent_touch_penalty":
                trial.params[
                    "opponent_touch_penalty"
                ],


            "block_penalty":
                trial.params[
                    "block_penalty"
                ],


            "chase_cooldown_steps":
                trial.params[
                    "chase_cooldown_steps"
                ],


            "agent_1_score":
                trial.user_attrs.get(
                    "agent_1_score",
                    np.nan
                ),


            "agent_2_score":
                trial.user_attrs.get(
                    "agent_2_score",
                    np.nan
                ),


            "agent_3_score":
                trial.user_attrs.get(
                    "agent_3_score",
                    np.nan
                ),


            "std_between_agents":
                trial.user_attrs.get(
                    "std_between_agents",
                    np.nan
                ),


            "average_red_goals":
                trial.user_attrs.get(
                    "average_red_goals",
                    np.nan
                ),


            "average_goal_difference":
                trial.user_attrs.get(
                    "average_goal_difference",
                    np.nan
                )
        })


    dataframe = pd.DataFrame(
        rows
    )


    dataframe.to_csv(

        CSV_PATH,

        index=False
    )


    return dataframe


# ================================================================
# GRAPHS
# ================================================================

def make_graphs(
    dataframe
):

    if dataframe.empty:

        return


    dataframe = dataframe.sort_values(
        "trial"
    )


    trials = dataframe[
        "trial"
    ]


    # ============================================================
    # MEDIAN VS MEAN
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        trials,

        dataframe[
            "median_blue_goals"
        ],

        marker="o",

        label="Median"
    )


    plt.plot(

        trials,

        dataframe[
            "average_blue_goals"
        ],

        marker="o",

        label="Mean"
    )


    plt.xlabel(
        "Trial"
    )


    plt.ylabel(
        "Blue goals / 60-second game"
    )


    plt.title(
        "Half-Speed Chaser Performance"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            OUTPUT_FOLDER,

            "01_median_vs_mean.png"
        ),

        dpi=200
    )


    plt.show()


    # ============================================================
    # INDIVIDUAL AGENTS
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )


    for number in [
        1,
        2,
        3
    ]:

        plt.plot(

            trials,

            dataframe[
                f"agent_{number}_score"
            ],

            marker="o",

            label=f"Agent {number}"
        )


    plt.xlabel(
        "Trial"
    )


    plt.ylabel(
        "Blue goals / game"
    )


    plt.title(
        "Individual Agent Performance"
    )


    plt.legend()


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            OUTPUT_FOLDER,

            "02_individual_agents.png"
        ),

        dpi=200
    )


    plt.show()


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)

    print(
        "OFFENSIVE AIR-HOCKEY "
        "HALF-SPEED DEFENSIVE CHASER"
    )

    print("=" * 70)


    print()
    print(
        f"Trials = "
        f"{TARGET_TRIALS}"
    )


    print(
        f"Agents per trial = "
        f"{AGENTS_PER_TRIAL}"
    )


    print(
        f"Additional timesteps per agent = "
        f"{CONTINUED_TIMESTEPS:,}"
    )


    print(
        f"Red speed = "
        f"{CHASER_SPEED_MULTIPLIER * 100:.0f}%"
    )


    print(
        "Optuna objective = MEDIAN"
    )


    print(
        f"Starting models = "
        f"V5 Trial {V5_STARTING_TRIAL}"
    )


    # ============================================================
    # CHECK STARTING MODELS
    # ============================================================

    print()
    print(
        "Checking V5 starting models..."
    )


    for seed in TRAINING_SEEDS:

        path = (
            get_starting_model_path(
                seed
            )
        )


        print(
            f"Seed {seed}: {path}"
        )


        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(

                f"\nMissing starting model:\n"
                f"{path}"
            )


    print(
        "All starting models found."
    )


    # ============================================================
    # CHECK ENVIRONMENT
    # ============================================================

    print()
    print(
        "Checking environment..."
    )


    test_env = make_env(

        BASELINE_HIT_REWARD,

        BASELINE_TOUCH_PENALTY,

        BASELINE_BLOCK_PENALTY,

        BASELINE_CHASE_COOLDOWN,

        training_mode=True
    )


    check_env(

        test_env,

        warn=True
    )


    test_env.close()


    print(
        "Environment check passed."
    )


    # ============================================================
    # OPTUNA
    # ============================================================

    sampler = (
        optuna.samplers.TPESampler(

            seed=24680,

            n_startup_trials=3
        )
    )


    storage_url = (

        "sqlite:///"

        +

        os.path.abspath(
            DATABASE_PATH
        )
    )


    study = optuna.create_study(

        study_name=STUDY_NAME,

        storage=storage_url,

        load_if_exists=True,

        direction="maximize",

        sampler=sampler
    )


    # ============================================================
    # TRIAL 1 = V5 WINNING VALUES
    # ============================================================

    if len(study.trials) == 0:

        study.enqueue_trial({

            "hit_reward":
                BASELINE_HIT_REWARD,

            "opponent_touch_penalty":
                BASELINE_TOUCH_PENALTY,

            "block_penalty":
                BASELINE_BLOCK_PENALTY,

            "chase_cooldown_steps":
                BASELINE_CHASE_COOLDOWN
        })


    # ============================================================
    # RESUME
    # ============================================================

    completed_trials = [

        trial

        for trial in study.trials

        if (
            trial.state
            ==
            optuna.trial.TrialState.COMPLETE
        )
    ]


    remaining = max(

        0,

        TARGET_TRIALS

        -

        len(completed_trials)
    )


    print()
    print(
        f"Completed trials: "
        f"{len(completed_trials)}"
    )


    print(
        f"Trials remaining: "
        f"{remaining}"
    )


    # ============================================================
    # RUN
    # ============================================================

    if remaining > 0:

        study.optimize(

            objective,

            n_trials=remaining
        )


    # ============================================================
    # RESULTS
    # ============================================================

    dataframe = (
        save_results_csv(
            study
        )
    )


    best_trial = (
        study.best_trial
    )


    print()
    print("=" * 70)

    print(
        "BEST HALF-SPEED CHASER TRIAL"
    )

    print("=" * 70)


    print(
        f"Trial: "
        f"{best_trial.number + 1}"
    )


    print(
        f"Median blue goals: "
        f"{best_trial.value:.3f}"
    )


    print(
        f"Mean blue goals: "
        f"{best_trial.user_attrs['average_blue_goals']:.3f}"
    )


    print()
    print(
        f"Hit reward: "
        f"{best_trial.params['hit_reward']:+.4f}"
    )


    print(
        f"Touch penalty: "
        f"{best_trial.params['opponent_touch_penalty']:+.4f}"
    )


    print(
        f"Block penalty: "
        f"{best_trial.params['block_penalty']:+.4f}"
    )


    print(
        f"Chase cooldown: "
        f"{best_trial.params['chase_cooldown_steps']} steps"
    )


    print()
    print(
        f"Agent 1: "
        f"{best_trial.user_attrs['agent_1_score']:.3f}"
    )


    print(
        f"Agent 2: "
        f"{best_trial.user_attrs['agent_2_score']:.3f}"
    )


    print(
        f"Agent 3: "
        f"{best_trial.user_attrs['agent_3_score']:.3f}"
    )


    print(
        f"Variation: "
        f"{best_trial.user_attrs['std_between_agents']:.3f}"
    )


    print(
        f"Average red goals: "
        f"{best_trial.user_attrs['average_red_goals']:.3f}"
    )


    print(
        f"Goal difference: "
        f"{best_trial.user_attrs['average_goal_difference']:+.3f}"
    )


    # ============================================================
    # COPY BEST MODELS
    # ============================================================

    best_number = (
        best_trial.number + 1
    )


    for seed in TRAINING_SEEDS:

        source = os.path.join(

            MODEL_FOLDER,

            (
                f"trial_{best_number}"
                f"_seed_{seed}.zip"
            )
        )


        destination = os.path.join(

            OUTPUT_FOLDER,

            (
                f"best_stage2_seed_{seed}.zip"
            )
        )


        if os.path.exists(
            source
        ):

            shutil.copyfile(

                source,

                destination
            )


    # ============================================================
    # GRAPHS
    # ============================================================

    make_graphs(
        dataframe
    )


    print()
    print("=" * 70)

    print(
        "FINISHED"
    )

    print("=" * 70)


    print(
        f"Results saved in: "
        f"{OUTPUT_FOLDER}"
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()


# %% WATCH HALF-SPEED DEFENSIVE CHASER AGENT

import os
import numpy as np
import pygame

from gymnasium import spaces
from stable_baselines3 import PPO

from air_hockey_env import AirHockeyEnv


# ================================================================
# CHOOSE WHICH AGENT TO WATCH
# ================================================================

WATCH_TRIAL = 5
WATCH_SEED = 1


# ================================================================
# GAME SETTINGS
# ================================================================

GAME_SECONDS = 60
FPS = 60

TOTAL_STEPS = (
    GAME_SECONDS
    *
    FPS
)


# ================================================================
# RED SPEED
# ================================================================

CHASER_SPEED_MULTIPLIER = 0.50


# ================================================================
# MODEL LOCATION
# ================================================================

MODEL_PATH = os.path.join(

    "offensive_vs_half_chaser_stage2",

    "models",

    f"trial_{WATCH_TRIAL}_seed_{WATCH_SEED}.zip"
)


print()
print("Loading model:")
print(MODEL_PATH)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(

        f"\nCould not find:\n"
        f"{MODEL_PATH}\n\n"
        f"Check that Trial {WATCH_TRIAL}, "
        f"Seed {WATCH_SEED} was trained."
    )


# ================================================================
# WATCH ENVIRONMENT
# ================================================================

class WatchHalfSpeedDefensiveChaserEnv(
    AirHockeyEnv
):

    def __init__(
        self,
        render_mode="human"
    ):

        super().__init__(
            render_mode=render_mode
        )


        # PPO controls BLUE only:
        #
        # [x movement, y movement]

        self.action_space = spaces.Box(

            low=-1.0,

            high=1.0,

            shape=(2,),

            dtype=np.float32
        )


    # ================================================================
    # RED DEFENSIVE ACTION
    # ================================================================

    def _red_defensive_action(
        self
    ):

        """
        RED behaviour:

        If puck is in RED'S half:
            chase the puck.

        If puck is in BLUE'S half:
            move back to the centre of RED'S half
            and wait there.
        """


        # ============================================================
        # PUCK IN RED'S HALF?
        # ============================================================

        puck_in_red_half = (

            self.puck_pos[1]

            <

            self.height / 2
        )


        # ============================================================
        # CHASE PUCK
        # ============================================================

        if puck_in_red_half:

            target_position = (
                self.puck_pos.copy()
            )


        # ============================================================
        # RETURN TO WAITING POSITION
        # ============================================================

        else:

            target_position = np.array(

                [
                    self.width / 2,
                    self.height * 0.25
                ],

                dtype=np.float32
            )


        # ============================================================
        # DIRECTION RED -> TARGET
        # ============================================================

        direction = (

            target_position

            -

            self.paddle2_pos
        )


        distance = float(

            np.linalg.norm(
                direction
            )
        )


        # ============================================================
        # ALREADY AT TARGET
        # ============================================================

        if distance <= 1e-8:

            return np.zeros(

                2,

                dtype=np.float32
            )


        # ============================================================
        # NORMALISE DIRECTION
        # ============================================================

        action = (

            direction

            /

            distance
        )


        return action.astype(
            np.float32
        )


    # ================================================================
    # POST-GOAL RESET
    # ================================================================

    def reset_after_goal(
        self,
        conceding_player
    ):

        # ------------------------------------------------------------
        # RESET BLUE
        # ------------------------------------------------------------

        self.paddle1_pos = np.array(

            [
                self.width / 2,
                self.height * 0.75
            ],

            dtype=np.float32
        )


        self.paddle1_vel = np.zeros(

            2,

            dtype=np.float32
        )


        # ------------------------------------------------------------
        # RESET RED
        # ------------------------------------------------------------

        self.paddle2_pos = np.array(

            [
                self.width / 2,
                self.height * 0.25
            ],

            dtype=np.float32
        )


        self.paddle2_vel = np.zeros(

            2,

            dtype=np.float32
        )


        # ------------------------------------------------------------
        # PUCK RESTART POSITION
        # ------------------------------------------------------------

        centre_y = (
            self.height / 2
        )


        one_third_of_half = (

            (self.height / 2)

            /

            3
        )


        # BLUE conceded.

        if conceding_player == 1:

            puck_y = (

                centre_y

                +

                one_third_of_half
            )


        # RED conceded.

        elif conceding_player == 2:

            puck_y = (

                centre_y

                -

                one_third_of_half
            )


        else:

            raise ValueError(
                "conceding_player must be 1 or 2"
            )


        self.puck_pos = np.array(

            [
                self.width / 2,
                puck_y
            ],

            dtype=np.float32
        )


        # Stationary restart.

        self.puck_vel = np.zeros(

            2,

            dtype=np.float32
        )


        return self._get_observation()


    # ================================================================
    # STEP
    # ================================================================

    def step(
        self,
        blue_action
    ):

        # ============================================================
        # BLUE ACTION
        # ============================================================

        blue_action = np.asarray(

            blue_action,

            dtype=np.float32
        )


        blue_action = np.clip(

            blue_action,

            -1.0,

            1.0
        )


        # ============================================================
        # RED ACTION
        # ============================================================

        red_action = (
            self._red_defensive_action()
        )


        # ============================================================
        # BLUE FULL SPEED
        # ============================================================

        self.paddle1_vel = (

            blue_action

            *

            self.paddle_speed
        )


        # ============================================================
        # RED 50% SPEED
        # ============================================================

        self.paddle2_vel = (

            red_action

            *

            self.paddle_speed

            *

            CHASER_SPEED_MULTIPLIER
        )


        # ============================================================
        # MOVE PADDLES
        # ============================================================

        self.paddle1_pos += (
            self.paddle1_vel
        )


        self.paddle2_pos += (
            self.paddle2_vel
        )


        self._limit_paddles()


        # ============================================================
        # MOVE PUCK
        # ============================================================

        self.puck_pos += (
            self.puck_vel
        )


        self._handle_wall_collisions()


        # ============================================================
        # PADDLE COLLISIONS
        # ============================================================

        self._handle_paddle_collision(

            self.paddle1_pos,

            self.paddle1_vel
        )


        self._handle_paddle_collision(

            self.paddle2_pos,

            self.paddle2_vel
        )


        # ============================================================
        # GOAL CHECK
        # ============================================================

        goal = (
            self._check_goal()
        )


        # ============================================================
        # BLUE SCORES
        # ============================================================

        if goal == 1:

            observation = (
                self.reset_after_goal(
                    conceding_player=2
                )
            )


        # ============================================================
        # RED SCORES
        # ============================================================

        elif goal == 2:

            observation = (
                self.reset_after_goal(
                    conceding_player=1
                )
            )


        # ============================================================
        # NO GOAL
        # ============================================================

        else:

            observation = (
                self._get_observation()
            )


        # Rewards do not matter when watching.

        reward = 0.0

        terminated = False

        truncated = False


        info = {
            "goal": goal
        }


        if self.render_mode == "human":

            self.render()


        return (

            observation,

            reward,

            terminated,

            truncated,

            info
        )


# ================================================================
# LOAD PPO MODEL
# ================================================================

model = PPO.load(
    MODEL_PATH
)


# ================================================================
# START PYGAME
# ================================================================

pygame.init()


env = WatchHalfSpeedDefensiveChaserEnv(
    render_mode="human"
)


observation, info = env.reset(
    seed=42
)


pygame.display.set_caption(

    f"Half-Speed Chaser - "
    f"Trial {WATCH_TRIAL} "
    f"- Seed {WATCH_SEED}"
)


pygame.event.pump()


# ================================================================
# SCORE
# ================================================================

blue_goals = 0

red_goals = 0


print()
print("=" * 60)

print(
    f"WATCHING TRIAL {WATCH_TRIAL} "
    f"| SEED {WATCH_SEED}"
)

print("=" * 60)

print(
    "BLUE = trained PPO agent"
)

print(
    "RED = 50% defensive chaser"
)

print(
    f"Game = {GAME_SECONDS} seconds"
)

print()


# ================================================================
# PLAY GAME
# ================================================================

for step in range(
    TOTAL_STEPS
):

    # ============================================================
    # KEEP PYGAME RESPONSIVE
    # ============================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            env.close()

            pygame.quit()

            raise SystemExit


    # ============================================================
    # BLUE CHOOSES ACTION
    # ============================================================

    action, _ = model.predict(

        observation,

        deterministic=True
    )


    # ============================================================
    # STEP ENVIRONMENT
    # ============================================================

    (
        observation,
        reward,
        terminated,
        truncated,
        info

    ) = env.step(
        action
    )


    # ============================================================
    # SCORE
    # ============================================================

    goal = info.get(
        "goal",
        None
    )


    if goal == 1:

        blue_goals += 1


        print(

            f"BLUE SCORES | "
            f"Blue {blue_goals} - "
            f"{red_goals} Red"
        )


    elif goal == 2:

        red_goals += 1


        print(

            f"RED SCORES  | "
            f"Blue {blue_goals} - "
            f"{red_goals} Red"
        )


# ================================================================
# FINISH
# ================================================================

env.close()

pygame.quit()


print()
print("=" * 60)

print(
    "FULL TIME"
)

print("=" * 60)


print(

    f"Trial {WATCH_TRIAL}, "
    f"Seed {WATCH_SEED}"
)


print(

    f"Final score: "
    f"Blue {blue_goals} - "
    f"{red_goals} Red"
)

print()