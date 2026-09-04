# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:35:29 2026

@author: nibra
"""

# -*- coding: utf-8 -*-

"""
OFFENSIVE AIR-HOCKEY REWARD TUNING V5
=====================================

BLUE = PPO offensive AI
RED  = random opponent


REWARDS
=======

Goal:
    +1.0 fixed

Hit puck:
    Tuned by Optuna

Useful chase start:
    +0.03 fixed

    This reward can ONLY activate when:

        1. The puck is in BLUE'S half.
        2. Blue is behind the puck relative to the red goal.
        3. Blue changes from a non-useful velocity
           into a useful chasing velocity.
        4. Blue's velocity is within 45 degrees
           of directly toward the puck.
        5. Chase cooldown has expired.

    The chase reward is NOT given continuously.

Chase cooldown:
    Tuned by Optuna from 5 to 60 steps.

Opponent touch:
    Negative reward tuned by Optuna.

Opponent block:
    Extra negative reward tuned by Optuna.


MATCH STRUCTURE
===============

Each training episode = one 60-second match.

START OF MATCH:
    puck begins normally in the centre.

BLUE SCORES:
    red conceded.
    puck restarts stationary in red's half.

RED SCORES:
    blue conceded.
    puck restarts stationary in blue's half.

A goal does NOT end the PPO episode.

Only reaching 60 simulated seconds ends the training episode.


OPTUNA
======

Every trial trains 3 completely separate PPO agents:

    Seed 1
    Seed 2
    Seed 3

Each agent trains for 200,000 timesteps.

Each agent is evaluated over:

    100 games
    60 seconds per game

Optuna maximises the average blue goals/game
across all three agents.
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

    TRAIN_TIMESTEPS = 10_000

    AGENTS_PER_TRIAL = 3

    EVALUATION_GAMES = 5

    EVALUATION_SECONDS = 10


else:

    TARGET_TRIALS = 5

    TRAIN_TIMESTEPS = 200_000

    AGENTS_PER_TRIAL = 3

    EVALUATION_GAMES = 100

    EVALUATION_SECONDS = 60


# ================================================================
# SIMULATION
# ================================================================

SIMULATION_FPS = 60


EVALUATION_STEPS_PER_GAME = (
    EVALUATION_SECONDS
    *
    SIMULATION_FPS
)


TRAIN_MATCH_SECONDS = 60


TRAIN_EPISODE_MAX_STEPS = (
    TRAIN_MATCH_SECONDS
    *
    SIMULATION_FPS
)


# ================================================================
# RANDOM OPPONENT
# ================================================================

OPPONENT_HOLD_STEPS = 15


# ================================================================
# TRAINING SEEDS
# ================================================================

TRAINING_SEEDS = [
    1,
    2,
    3
]


# ================================================================
# EVALUATION SEEDS
# ================================================================

EVALUATION_SEED_START = 10_000


# ================================================================
# FIXED REWARDS
# ================================================================

GOAL_REWARD = 1.0


CHASE_START_REWARD = 0.03


# ================================================================
# CHASE DIRECTION REQUIREMENT
# ================================================================

# EXTRA DESIGN CHOICE:
#
# You asked for blue to actually be pointing towards the puck,
# but did not specify an exact angle.
#
# 45 degrees means blue's velocity must lie within a
# 90-degree-wide cone centred directly on the puck.

CHASE_MAX_ANGLE_DEGREES = 45


# ================================================================
# BASELINE REWARD SYSTEM
# ================================================================

ORIGINAL_HIT_REWARD = 0.05

ORIGINAL_TOUCH_PENALTY = -0.10

ORIGINAL_BLOCK_PENALTY = -0.30

ORIGINAL_CHASE_COOLDOWN = 15


# ================================================================
# OPTUNA SEARCH RANGES
# ================================================================

HIT_REWARD_MIN = 0.0
HIT_REWARD_MAX = 0.25


TOUCH_PENALTY_MIN = -0.75
TOUCH_PENALTY_MAX = 0.0


BLOCK_PENALTY_MIN = -1.50
BLOCK_PENALTY_MAX = 0.0


# 5 steps  = 0.083 seconds
# 15 steps = 0.25 seconds
# 30 steps = 0.50 seconds
# 60 steps = 1.00 second

CHASE_COOLDOWN_MIN = 5
CHASE_COOLDOWN_MAX = 60


# ================================================================
# OUTPUT
# ================================================================

# NEW V5 FOLDER/DATABASE because chase reward logic has changed.

if QUICK_TEST:

    OUTPUT_FOLDER = (
        "offensive_reward_tuning_v5_test"
    )

else:

    OUTPUT_FOLDER = (
        "offensive_reward_tuning_v5"
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
    "offensive_air_hockey_reward_tuning_v5"
)


# ================================================================
# OFFENSIVE ENVIRONMENT
# ================================================================

class OffensiveAirHockeyEnv(AirHockeyEnv):

    def __init__(
        self,
        hit_reward=0.05,
        opponent_touch_penalty=-0.10,
        block_penalty=-0.30,
        chase_cooldown_steps=15,
        render_mode=None,
        training_mode=True
    ):

        super().__init__(
            render_mode=render_mode
        )


        # ============================================================
        # REWARDS
        # ============================================================

        self.goal_reward = GOAL_REWARD


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
        # MATCH TIMER
        # ============================================================

        self.episode_steps = 0


        # ============================================================
        # SHOT TRACKING
        # ============================================================

        self.ai_shot_active = False


        # ============================================================
        # CHASE COOLDOWN
        # ============================================================

        self.chase_reward_cooldown = 0


        # ============================================================
        # RANDOM RED OPPONENT
        # ============================================================

        self.opponent_action = np.zeros(
            2,
            dtype=np.float32
        )


        self.opponent_steps_remaining = 0


    # ================================================================
    # NORMAL RESET
    # ================================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        """
        Starts a completely new 60-second match.

        Original AirHockeyEnv reset is used, so the
        puck begins normally in the centre.
        """

        observation, info = super().reset(
            seed=seed,
            options=options
        )


        self.episode_steps = 0


        self.ai_shot_active = False


        self.chase_reward_cooldown = 0


        self.opponent_action = np.zeros(
            2,
            dtype=np.float32
        )


        self.opponent_steps_remaining = 0


        return observation, info


    # ================================================================
    # RESET AFTER GOAL
    # ================================================================

    def reset_after_goal(
        self,
        conceding_player
    ):

        """
        Restart play without resetting the 60-second timer.

        conceding_player = 1
            BLUE conceded.
            Puck starts in blue's half.

        conceding_player = 2
            RED conceded.
            Puck starts in red's half.
        """


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
        # PUCK RESTART POSITION
        # ============================================================

        centre_y = (
            self.height / 2
        )


        half_field_length = (
            self.height / 2
        )


        one_third_of_half = (
            half_field_length
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


        self.ai_shot_active = False


        # A new restart can immediately generate a new chase event.

        self.chase_reward_cooldown = 0


        self.opponent_action = np.zeros(
            2,
            dtype=np.float32
        )


        self.opponent_steps_remaining = 0


        observation = (
            self._get_observation()
        )


        return observation


    # ================================================================
    # RANDOM OPPONENT ACTION
    # ================================================================

    def _random_opponent_action(
        self
    ):

        if self.opponent_steps_remaining <= 0:

            self.opponent_action = (
                self.np_random.uniform(

                    low=-1.0,

                    high=1.0,

                    size=2

                ).astype(np.float32)
            )


            self.opponent_steps_remaining = (
                OPPONENT_HOLD_STEPS
            )


        self.opponent_steps_remaining -= 1


        return (
            self.opponent_action.copy()
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
    # IS VELOCITY A USEFUL CHASE VELOCITY?
    # ================================================================

    def _is_useful_chase_velocity(
        self,
        velocity
    ):

        """
        Chase only counts when:

        1. Puck is in BLUE'S half.
        2. Blue is behind the puck.
        3. Blue is actually moving.
        4. Blue's velocity is within 45 degrees
           of directly toward the puck.

        Blue attacks toward y = 0.
        Therefore blue's half is y > height / 2.
        """


        # ============================================================
        # 1. PUCK MUST BE IN BLUE'S HALF
        # ============================================================

        puck_in_blue_half = (

            self.puck_pos[1]

            >

            self.height / 2
        )


        if not puck_in_blue_half:

            return False


        # ============================================================
        # 2. BLUE MUST BE BEHIND PUCK
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


        distance_to_puck = float(

            np.linalg.norm(
                to_puck
            )
        )


        if distance_to_puck <= 1e-8:

            return False


        # ============================================================
        # BLUE MOVEMENT SPEED
        # ============================================================

        speed = float(

            np.linalg.norm(
                velocity
            )
        )


        if speed <= 1e-8:

            return False


        # ============================================================
        # COSINE OF ANGLE BETWEEN VELOCITY AND PUCK DIRECTION
        # ============================================================

        cosine_of_angle = float(

            np.dot(

                velocity,

                to_puck
            )

            /

            (
                speed

                *

                distance_to_puck
            )
        )


        cosine_of_angle = float(

            np.clip(

                cosine_of_angle,

                -1.0,

                1.0
            )
        )


        # ============================================================
        # MAXIMUM ALLOWED ANGLE
        # ============================================================

        maximum_angle_radians = (
            np.deg2rad(
                CHASE_MAX_ANGLE_DEGREES
            )
        )


        minimum_allowed_cosine = float(

            np.cos(
                maximum_angle_radians
            )
        )


        # ============================================================
        # USEFUL CHASE?
        # ============================================================

        return (

            cosine_of_angle

            >=

            minimum_allowed_cosine
        )


    # ================================================================
    # COLLISION + TRUE HIT DETECTION
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
        # TRUE HIT CHECK
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
            relative_speed < 0
        )


        # ============================================================
        # ORIGINAL COLLISION PHYSICS
        # ============================================================

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
        ai_action
    ):

        self.episode_steps += 1


        # ============================================================
        # BLUE ACTION
        # ============================================================

        ai_action = np.asarray(

            ai_action,

            dtype=np.float32
        )


        ai_action = np.clip(

            ai_action,

            -1.0,

            1.0
        )


        # ============================================================
        # SAVE OLD BLUE VELOCITY
        # ============================================================

        old_blue_velocity = (
            self.paddle1_vel.copy()
        )


        # ============================================================
        # NEW BLUE VELOCITY
        # ============================================================

        new_blue_velocity = (

            ai_action

            *

            self.paddle_speed
        )


        # ============================================================
        # RANDOM RED
        # ============================================================

        opponent_action = (
            self._random_opponent_action()
        )


        new_red_velocity = (

            opponent_action

            *

            self.paddle_speed
        )


        reward = 0.0


        # ============================================================
        # CHASE COOLDOWN
        # ============================================================

        if self.chase_reward_cooldown > 0:

            self.chase_reward_cooldown -= 1


        # ============================================================
        # OLD AND NEW CHASE DIRECTIONS
        # ============================================================

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


        # ============================================================
        # DID VELOCITY ACTUALLY CHANGE?
        # ============================================================

        velocity_change = float(

            np.linalg.norm(

                new_blue_velocity

                -

                old_blue_velocity
            )
        )


        velocity_changed = (
            velocity_change > 1e-6
        )


        # ============================================================
        # CHANGED INTO A USEFUL CHASE?
        # ============================================================

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
        # APPLY PADDLE VELOCITIES
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
        # BLUE HITS PUCK
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
        # ACTIVE SHOT HEADING TOWARD GOAL?
        # ============================================================

        was_heading_towards_goal = (

            self.ai_shot_active

            and

            self._puck_heading_towards_goal()
        )


        # ============================================================
        # RED HITS PUCK
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
        # GOAL CHECK
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
            #
            # This is still the offensive specialist.

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
        # END OF TRAINING MATCH
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

            "old_velocity_useful":
                old_velocity_useful,

            "new_velocity_useful":
                new_velocity_useful,

            "started_useful_chase":
                started_useful_chase,

            "chase_cooldown_remaining":
                self.chase_reward_cooldown,

            "goal_reward":
                GOAL_REWARD,

            "hit_reward":
                self.hit_reward,

            "chase_start_reward":
                CHASE_START_REWARD,

            "chase_max_angle_degrees":
                CHASE_MAX_ANGLE_DEGREES,

            "chase_cooldown_steps":
                self.chase_cooldown_steps,

            "opponent_touch_penalty":
                self.opponent_touch_penalty,

            "block_penalty":
                self.block_penalty
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
# MAKE ENVIRONMENT
# ================================================================

def make_env(
    hit_reward,
    opponent_touch_penalty,
    block_penalty,
    chase_cooldown_steps,
    training_mode=True,
    render_mode=None
):

    return OffensiveAirHockeyEnv(

        hit_reward=hit_reward,

        opponent_touch_penalty=opponent_touch_penalty,

        block_penalty=block_penalty,

        chase_cooldown_steps=chase_cooldown_steps,

        render_mode=render_mode,

        training_mode=training_mode
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


    blue_goals_each_game = []

    red_goals_each_game = []


    for game_number in range(
        EVALUATION_GAMES
    ):

        evaluation_seed = (

            EVALUATION_SEED_START

            +

            game_number
        )


        observation, info = env.reset(
            seed=evaluation_seed
        )


        blue_goals = 0

        red_goals = 0


        for step in range(
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


        blue_goals_each_game.append(
            blue_goals
        )


        red_goals_each_game.append(
            red_goals
        )


    env.close()


    blue_goals_each_game = np.array(

        blue_goals_each_game,

        dtype=np.float32
    )


    red_goals_each_game = np.array(

        red_goals_each_game,

        dtype=np.float32
    )


    return {

        "average_blue_goals":

            float(
                np.mean(
                    blue_goals_each_game
                )
            ),


        "std_blue_goals":

            float(
                np.std(
                    blue_goals_each_game
                )
            ),


        "average_red_goals":

            float(
                np.mean(
                    red_goals_each_game
                )
            ),


        "average_goal_difference":

            float(
                np.mean(

                    blue_goals_each_game

                    -

                    red_goals_each_game
                )
            ),


        "total_blue_goals":

            int(
                np.sum(
                    blue_goals_each_game
                )
            ),


        "total_red_goals":

            int(
                np.sum(
                    red_goals_each_game
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
        f"OPTUNA TRIAL "
        f"{trial.number + 1}"
    )

    print("=" * 70)


    # ============================================================
    # OPTUNA PARAMETERS
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
    print(
        "Reward system:"
    )


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


    # ============================================================
    # RESULTS FROM THREE AGENTS
    # ============================================================

    agent_scores = []

    agent_red_scores = []

    agent_goal_differences = []


    # ============================================================
    # TRAIN THREE AGENTS
    # ============================================================

    for agent_index, seed in enumerate(

        TRAINING_SEEDS,

        start=1
    ):

        print()

        print(
            f"--- Agent "
            f"{agent_index}/"
            f"{AGENTS_PER_TRIAL} "
            f"| seed {seed} ---"
        )


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


        model = PPO(

            "MlpPolicy",

            env,

            seed=seed,

            verbose=0,

            device="auto"
        )


        print(
            f"Training "
            f"{TRAIN_TIMESTEPS:,} "
            f"timesteps..."
        )


        model.learn(
            total_timesteps=TRAIN_TIMESTEPS
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
    # TRIAL AVERAGES
    # ============================================================

    average_blue_goals = float(

        np.mean(
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
    # PRINT TRIAL RESULT
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
        f"Average across 3 agents: "
        f"{average_blue_goals:.3f}"
    )


    print(
        f"Variation between agents: "
        f"{std_between_agents:.3f}"
    )


    print(
        f"Average red goals: "
        f"{average_red_goals:.3f}"
    )


    print(
        f"Average goal difference: "
        f"{average_goal_difference:+.3f}"
    )


    # ============================================================
    # STORE OPTUNA INFORMATION
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


    return average_blue_goals


# ================================================================
# SAVE RESULTS CSV
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


            "average_blue_goals":
                trial.value,


            "goal_reward":
                GOAL_REWARD,


            "chase_start_reward":
                CHASE_START_REWARD,


            "chase_max_angle_degrees":
                CHASE_MAX_ANGLE_DEGREES,


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
    ].to_numpy()


    # ============================================================
    # GRAPH 1 - REWARD VALUES
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        trials,

        dataframe[
            "hit_reward"
        ],

        marker="o",

        label="Hit reward"
    )


    plt.plot(

        trials,

        dataframe[
            "opponent_touch_penalty"
        ],

        marker="o",

        label="Opponent touch"
    )


    plt.plot(

        trials,

        dataframe[
            "block_penalty"
        ],

        marker="o",

        label="Block penalty"
    )


    plt.xlabel(
        "Optuna trial"
    )


    plt.ylabel(
        "Reward value"
    )


    plt.title(
        "Reward Values Tested"
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
            "01_reward_values.png"
        ),

        dpi=200
    )


    plt.show()


    # ============================================================
    # GRAPH 2 - CHASE COOLDOWN
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        trials,

        dataframe[
            "chase_cooldown_steps"
        ],

        marker="o"
    )


    plt.xlabel(
        "Optuna trial"
    )


    plt.ylabel(
        "Chase cooldown (steps)"
    )


    plt.title(
        "Chase Cooldown Tested"
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(
            OUTPUT_FOLDER,
            "02_chase_cooldown.png"
        ),

        dpi=200
    )


    plt.show()


    # ============================================================
    # GRAPH 3 - PERFORMANCE
    # ============================================================

    scores = dataframe[
        "average_blue_goals"
    ].to_numpy()


    best_so_far = (
        np.maximum.accumulate(
            scores
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        trials,

        scores,

        marker="o",

        label="3-agent average"
    )


    plt.plot(

        trials,

        best_so_far,

        marker="o",

        label="Best so far"
    )


    plt.xlabel(
        "Optuna trial"
    )


    plt.ylabel(
        "Average blue goals / 60-second game"
    )


    plt.title(
        "Reward Tuning Performance"
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
            "03_performance.png"
        ),

        dpi=200
    )


    plt.show()


    # ============================================================
    # GRAPH 4 - INDIVIDUAL SEEDS
    # ============================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        trials,

        dataframe[
            "agent_1_score"
        ],

        marker="o",

        label="Seed 1"
    )


    plt.plot(

        trials,

        dataframe[
            "agent_2_score"
        ],

        marker="o",

        label="Seed 2"
    )


    plt.plot(

        trials,

        dataframe[
            "agent_3_score"
        ],

        marker="o",

        label="Seed 3"
    )


    plt.xlabel(
        "Optuna trial"
    )


    plt.ylabel(
        "Average blue goals/game"
    )


    plt.title(
        "Individual Training Seeds"
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
            "04_seed_results.png"
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
        "REWARD TUNING V5"
    )

    print("=" * 70)


    print()
    print(
        f"QUICK_TEST = "
        f"{QUICK_TEST}"
    )


    print(
        f"Trials = "
        f"{TARGET_TRIALS}"
    )


    print(
        f"Agents per trial = "
        f"{AGENTS_PER_TRIAL}"
    )


    print(
        f"Training timesteps per agent = "
        f"{TRAIN_TIMESTEPS:,}"
    )


    print(
        f"Training match length = "
        f"{TRAIN_MATCH_SECONDS} seconds"
    )


    print(
        f"Evaluation games per agent = "
        f"{EVALUATION_GAMES}"
    )


    print(
        f"Evaluation game length = "
        f"{EVALUATION_SECONDS} seconds"
    )


    print()
    print(
        f"Goal reward = "
        f"{GOAL_REWARD:+.3f}"
    )


    print(
        f"Chase-start reward = "
        f"{CHASE_START_REWARD:+.3f}"
    )


    print(
        f"Chase max angle = "
        f"{CHASE_MAX_ANGLE_DEGREES} degrees"
    )


    print(
        f"Chase cooldown search = "
        f"{CHASE_COOLDOWN_MIN}"
        f"–"
        f"{CHASE_COOLDOWN_MAX} steps"
    )


    # ============================================================
    # CHECK ENVIRONMENT
    # ============================================================

    print()
    print(
        "Checking environment..."
    )


    test_env = make_env(

        ORIGINAL_HIT_REWARD,

        ORIGINAL_TOUCH_PENALTY,

        ORIGINAL_BLOCK_PENALTY,

        ORIGINAL_CHASE_COOLDOWN,

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
    # OPTUNA SAMPLER
    # ============================================================

    startup_trials = min(

        10,

        max(
            3,
            TARGET_TRIALS // 5
        )
    )


    sampler = (
        optuna.samplers.TPESampler(

            seed=12345,

            n_startup_trials=startup_trials
        )
    )


    # ============================================================
    # DATABASE
    # ============================================================

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
    # TRIAL 1 = BASELINE
    # ============================================================

    if len(study.trials) == 0:

        study.enqueue_trial({

            "hit_reward":
                ORIGINAL_HIT_REWARD,

            "opponent_touch_penalty":
                ORIGINAL_TOUCH_PENALTY,

            "block_penalty":
                ORIGINAL_BLOCK_PENALTY,

            "chase_cooldown_steps":
                ORIGINAL_CHASE_COOLDOWN
        })


    # ============================================================
    # RESUME SUPPORT
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


    trials_remaining = max(

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
        f"{trials_remaining}"
    )


    # ============================================================
    # RUN OPTUNA
    # ============================================================

    if trials_remaining > 0:

        study.optimize(

            objective,

            n_trials=trials_remaining
        )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    dataframe = (
        save_results_csv(
            study
        )
    )


    best_trial = (
        study.best_trial
    )


    # ============================================================
    # BEST RESULT
    # ============================================================

    print()
    print()
    print("=" * 70)

    print(
        "BEST REWARD SYSTEM FOUND"
    )

    print("=" * 70)


    print()
    print(
        f"Trial: "
        f"{best_trial.number + 1}"
    )


    print(
        f"Three-agent average: "
        f"{best_trial.value:.3f}"
    )


    print()
    print(
        f"Goal reward:       "
        f"{GOAL_REWARD:+.4f}"
    )


    print(
        f"Hit reward:        "
        f"{best_trial.params['hit_reward']:+.4f}"
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
        f"{best_trial.params['chase_cooldown_steps']} steps"
    )


    print(
        f"Opponent touch:    "
        f"{best_trial.params['opponent_touch_penalty']:+.4f}"
    )


    print(
        f"Block penalty:     "
        f"{best_trial.params['block_penalty']:+.4f}"
    )


    print()
    print(
        "Individual seed scores:"
    )


    print(
        f"Seed 1: "
        f"{best_trial.user_attrs.get('agent_1_score', np.nan):.3f}"
    )


    print(
        f"Seed 2: "
        f"{best_trial.user_attrs.get('agent_2_score', np.nan):.3f}"
    )


    print(
        f"Seed 3: "
        f"{best_trial.user_attrs.get('agent_3_score', np.nan):.3f}"
    )


    print(
        f"Variation between agents: "
        f"{best_trial.user_attrs.get('std_between_agents', np.nan):.3f}"
    )


    print(
        f"Average red goals: "
        f"{best_trial.user_attrs.get('average_red_goals', np.nan):.3f}"
    )


    print(
        f"Average goal difference: "
        f"{best_trial.user_attrs.get('average_goal_difference', np.nan):+.3f}"
    )


    # ============================================================
    # COPY BEST MODELS
    # ============================================================

    best_trial_number = (
        best_trial.number + 1
    )


    for seed in TRAINING_SEEDS:

        source = os.path.join(

            MODEL_FOLDER,

            (
                f"trial_{best_trial_number}"
                f"_seed_{seed}.zip"
            )
        )


        destination = os.path.join(

            OUTPUT_FOLDER,

            (
                f"best_model_seed_{seed}.zip"
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


    # ============================================================
    # FINISH
    # ============================================================

    print()
    print("=" * 70)

    print(
        "FINISHED"
    )

    print("=" * 70)


    print()
    print(
        "Results saved in:"
    )


    print(
        OUTPUT_FOLDER
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()


# %% WATCH ANY V5 TRIAL + SEED


# ================================================================
# IMPORTS
# ================================================================

import os
import pandas as pd
import pygame

from stable_baselines3 import PPO


# ================================================================
# CHOOSE AGENT
# ================================================================

WATCH_TRIAL = 3

WATCH_SEED = 1


# Leave fixed if you want different agents to face
# the same random opponent sequence.

OPPONENT_SEED = 42


# ================================================================
# GAME LENGTH
# ================================================================

GAME_SECONDS = 60

FPS = 60


TOTAL_STEPS = (
    GAME_SECONDS
    *
    FPS
)


# ================================================================
# FILE LOCATIONS
# ================================================================

WATCH_OUTPUT_FOLDER = (
    "offensive_reward_tuning_v5"
)


MODEL_PATH = os.path.join(

    WATCH_OUTPUT_FOLDER,

    "models",

    f"trial_{WATCH_TRIAL}_seed_{WATCH_SEED}.zip"
)


RESULTS_PATH = os.path.join(

    WATCH_OUTPUT_FOLDER,

    "all_trials.csv"
)


print()
print(
    "Loading model:"
)

print(
    MODEL_PATH
)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        f"\nCould not find:\n"
        f"{MODEL_PATH}\n\n"
        f"Check Trial {WATCH_TRIAL}, "
        f"Seed {WATCH_SEED}."
    )


# ================================================================
# LOAD THAT TRIAL'S PARAMETERS
# ================================================================

results = pd.read_csv(
    RESULTS_PATH
)


trial_row = results[

    results["trial"]

    ==

    WATCH_TRIAL
]


if trial_row.empty:

    raise ValueError(

        f"Trial {WATCH_TRIAL} "
        f"was not found in all_trials.csv"
    )


trial_row = (
    trial_row.iloc[0]
)


hit_reward = float(
    trial_row["hit_reward"]
)


opponent_touch_penalty = float(
    trial_row["opponent_touch_penalty"]
)


block_penalty = float(
    trial_row["block_penalty"]
)


chase_cooldown_steps = int(
    trial_row["chase_cooldown_steps"]
)


# ================================================================
# LOAD MODEL
# ================================================================

model = PPO.load(
    MODEL_PATH
)


# ================================================================
# START PYGAME
# ================================================================

pygame.init()


env = make_env(

    hit_reward,

    opponent_touch_penalty,

    block_penalty,

    chase_cooldown_steps,

    training_mode=False,

    render_mode="human"
)


observation, info = env.reset(
    seed=OPPONENT_SEED
)


pygame.display.set_caption(

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
    "BLUE = trained offensive AI"
)


print(
    "RED  = random opponent"
)


print(
    f"Game = {GAME_SECONDS} seconds"
)


print()


# ================================================================
# PLAY
# ================================================================

for step in range(
    TOTAL_STEPS
):

    # ============================================================
    # PYGAME EVENTS
    # ============================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            env.close()

            pygame.quit()

            raise SystemExit


    # ============================================================
    # BLUE ACTION
    # ============================================================

    action, _ = model.predict(

        observation,

        deterministic=True
    )


    # ============================================================
    # ENVIRONMENT STEP
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
    # COUNT GOALS
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


    # IMPORTANT:
    #
    # Do NOT manually reset after goals.
    #
    # env.step() already uses the same conceding-side
    # restart system that was used during training.


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
    
# %% Watch any trial + seed for a 60-second game

import os
import numpy as np
import pygame

from gymnasium import spaces
from stable_baselines3 import PPO

from air_hockey_env import AirHockeyEnv


# ================================================================
# CHOOSE WHICH AGENT TO WATCH
# ================================================================

# Change these two numbers whenever you want to watch
# a different saved agent.

WATCH_TRIAL = 3
WATCH_SEED = 1


# Random opponent seed.
#
# Leave this fixed if you want different agents to face
# exactly the same random opponent behaviour when watching.

OPPONENT_SEED = 42


# ================================================================
# GAME LENGTH
# ================================================================

GAME_SECONDS = 60
FPS = 60

TOTAL_STEPS = (
    GAME_SECONDS *
    FPS
)


# ================================================================
# MODEL LOCATION
# ================================================================

MODEL_PATH = os.path.join(
    "offensive_reward_tuning_v4",
    "models",
    f"trial_{WATCH_TRIAL}_seed_{WATCH_SEED}.zip"
)


print()
print("Loading model:")
print(MODEL_PATH)


# Check that the model actually exists.

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nCould not find:\n{MODEL_PATH}\n\n"
        f"Check that Trial {WATCH_TRIAL}, "
        f"Seed {WATCH_SEED} was actually trained."
    )


# ================================================================
# SIMPLE VISUAL ENVIRONMENT
# ================================================================

class WatchOffensiveEnv(AirHockeyEnv):

    def __init__(self, render_mode="human"):

        super().__init__(
            render_mode=render_mode
        )


        # PPO controls blue only:
        #
        # [x movement, y movement]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )


        # Random red opponent.

        self.opponent_action = np.zeros(
            2,
            dtype=np.float32
        )

        self.opponent_steps_remaining = 0


        # IMPORTANT:
        # This is the same value used during training.

        self.opponent_hold_steps = 15


    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        observation, info = super().reset(
            seed=seed,
            options=options
        )


        self.opponent_action = np.zeros(
            2,
            dtype=np.float32
        )

        self.opponent_steps_remaining = 0


        return observation, info


    # ============================================================
    # RANDOM OPPONENT
    # ============================================================

    def _random_opponent_action(self):

        if self.opponent_steps_remaining <= 0:

            self.opponent_action = (
                self.np_random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=2
                )
                .astype(np.float32)
            )


            self.opponent_steps_remaining = (
                self.opponent_hold_steps
            )


        self.opponent_steps_remaining -= 1


        return self.opponent_action.copy()


    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        blue_action
    ):

        # --------------------------------------------------------
        # BLUE = TRAINED PPO
        # --------------------------------------------------------

        blue_action = np.asarray(
            blue_action,
            dtype=np.float32
        )


        blue_action = np.clip(
            blue_action,
            -1.0,
            1.0
        )


        # --------------------------------------------------------
        # RED = RANDOM
        # --------------------------------------------------------

        red_action = (
            self._random_opponent_action()
        )


        # --------------------------------------------------------
        # Original AirHockeyEnv expects:
        #
        # [
        #   blue_x,
        #   blue_y,
        #   red_x,
        #   red_y
        # ]
        # --------------------------------------------------------

        combined_action = np.concatenate(
            [
                blue_action,
                red_action
            ]
        )


        return super().step(
            combined_action
        )


# ================================================================
# LOAD THE CHOSEN PPO
# ================================================================

model = PPO.load(
    MODEL_PATH
)


# ================================================================
# START PYGAME
# ================================================================

pygame.init()


env = WatchOffensiveEnv(
    render_mode="human"
)


observation, info = env.reset(
    seed=OPPONENT_SEED
)


pygame.display.set_caption(
    f"Trial {WATCH_TRIAL} - Seed {WATCH_SEED}"
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
    "BLUE = trained offensive AI"
)

print(
    "RED  = random opponent"
)

print(
    f"Game = {GAME_SECONDS} seconds"
)

print()


# ================================================================
# PLAY FOR 60 SECONDS
# ================================================================

for step in range(
    TOTAL_STEPS
):

    # ------------------------------------------------------------
    # Allow the Pygame window to respond properly.
    # ------------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            env.close()

            pygame.quit()

            raise SystemExit


    # ------------------------------------------------------------
    # BLUE CHOOSES ACTION
    # ------------------------------------------------------------

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


    # ============================================================
    # GOAL
    # ============================================================

    goal = info.get(
        "goal",
        None
    )


    # ------------------------------------------------------------
    # BLUE SCORES
    # ------------------------------------------------------------

    if goal == 1:

        blue_goals += 1


        print(
            f"BLUE SCORES | "
            f"Blue {blue_goals} - "
            f"{red_goals} Red"
        )


        # Reset table but continue the SAME 60-second game.

        observation, info = env.reset()


    # ------------------------------------------------------------
    # RED SCORES
    # ------------------------------------------------------------

    elif goal == 2:

        red_goals += 1


        print(
            f"RED SCORES  | "
            f"Blue {blue_goals} - "
            f"{red_goals} Red"
        )


        observation, info = env.reset()


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

