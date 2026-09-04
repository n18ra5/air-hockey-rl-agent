# -*- coding: utf-8 -*-
"""
3 REWARDS - FULL SPEED CHASER
=============================

BLUE:
    Fresh PPO agent trained from scratch.

RED:
    100%-speed chaser bot.

ONLY THREE REWARDS
==================
Optuna tunes all three:

    Blue scores:       +0.5 to +2.0
    Blue concedes:     -2.0 to -0.5
    Blue hits puck:    +0.05 to +1.0

The puck-hit reward is a SINGLE-EVENT reward:
it is paid only on a genuine incoming paddle/puck collision.

There are NO:
    approach rewards
    movement penalties
    save rewards
    block rewards
    chase rewards
    positioning rewards
    wall rewards

TRAINING
========
5 Optuna trials
3 seeds per trial: 1, 2, 3
500,000 timesteps per seed
All agents start completely from scratch.

EVALUATION
==========
After training, every seed is evaluated over:
    100 games
    60 seconds per game

Optuna objective:
    MAXIMISE the MEDIAN goal difference across the 3 seeds.

Goal difference:
    average BLUE goals/game - average RED goals/game

OUTPUT
======
Training is silent.
Only post-training evaluation results are printed.

Models/results are saved under:
    3rewards_full_chaser/
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import numpy as np
import pandas as pd
import optuna

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from air_hockey_env import AirHockeyEnv


# ================================================================
# SETTINGS
# ================================================================

TRAIN_TIMESTEPS = 500_000
TRAINING_SEEDS = [1, 2, 3]

N_TRIALS = 5

SIMULATION_FPS = 60
TRAIN_MATCH_SECONDS = 60
TRAIN_EPISODE_MAX_STEPS = (
    TRAIN_MATCH_SECONDS * SIMULATION_FPS
)

EVALUATION_GAMES = 100
EVALUATION_SECONDS = 60
EVALUATION_STEPS_PER_GAME = (
    EVALUATION_SECONDS * SIMULATION_FPS
)

CHASER_SPEED_FRACTION = 1.00


# ================================================================
# OPTUNA REWARD RANGES
# ================================================================

GOAL_REWARD_MIN = 0.5
GOAL_REWARD_MAX = 2.0

CONCEDE_PENALTY_MIN = -2.0
CONCEDE_PENALTY_MAX = -0.5

HIT_REWARD_MIN = 0.05
HIT_REWARD_MAX = 1.0


# ================================================================
# RANDOMISED START / RESTART POSITIONS
# ================================================================
# Carried over from the recent full-chaser training setup.

RANDOM_X_MIN_FRACTION = 0.20
RANDOM_X_MAX_FRACTION = 0.80

BLUE_Y_MIN_FRACTION = 0.60
BLUE_Y_MAX_FRACTION = 0.90

RED_Y_MIN_FRACTION = 0.10
RED_Y_MAX_FRACTION = 0.40

RESET_PUCK_Y_MIN_FRACTION = 0.35
RESET_PUCK_Y_MAX_FRACTION = 0.65

PUCK_HALF_MARGIN_FRACTION = 0.08


# ================================================================
# PUCK STALL RESET
# ================================================================
# Preserves the existing rule:
# if puck speed stays below 10% of max puck speed for 1.5 seconds,
# reset it to the centre.

PUCK_STALL_SECONDS = 1.5
PUCK_STALL_STEPS = int(
    PUCK_STALL_SECONDS * SIMULATION_FPS
)

PUCK_STALL_SPEED_FRACTION = 0.10


# ================================================================
# OUTPUT
# ================================================================

OUTPUT_FOLDER = "3rewards_full_chaser"
MODEL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "models"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

CSV_PATH = os.path.join(
    OUTPUT_FOLDER,
    "all_trial_results.csv"
)

OPTUNA_DB_PATH = os.path.join(
    OUTPUT_FOLDER,
    "3rewards_full_chaser_optuna.db"
)

OPTUNA_STORAGE = (
    "sqlite:///"
    + os.path.abspath(OPTUNA_DB_PATH)
)

STUDY_NAME = "3rewards_full_chaser_goal_difference"


# ================================================================
# TRAINING ENVIRONMENT
# ================================================================

class ThreeRewardFullChaserEnv(AirHockeyEnv):

    def __init__(
        self,
        render_mode=None,
        training_mode=True,
        goal_reward=1.0,
        concede_penalty=-1.0,
        hit_reward=0.5
    ):

        super().__init__(
            render_mode=render_mode
        )

        self.training_mode = bool(
            training_mode
        )

        self.goal_reward = float(
            goal_reward
        )

        self.concede_penalty = float(
            concede_penalty
        )

        self.hit_reward = float(
            hit_reward
        )

        # PPO controls BLUE only.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.episode_steps = 0
        self.puck_stationary_steps = 0


    # ------------------------------------------------------------
    # Random position helpers
    # ------------------------------------------------------------

    def _random_x(self):

        return float(
            self.np_random.uniform(
                self.width
                * RANDOM_X_MIN_FRACTION,
                self.width
                * RANDOM_X_MAX_FRACTION
            )
        )


    def _random_blue_y(self):

        return float(
            self.np_random.uniform(
                self.height
                * BLUE_Y_MIN_FRACTION,
                self.height
                * BLUE_Y_MAX_FRACTION
            )
        )


    def _random_red_y(self):

        return float(
            self.np_random.uniform(
                self.height
                * RED_Y_MIN_FRACTION,
                self.height
                * RED_Y_MAX_FRACTION
            )
        )


    # ------------------------------------------------------------
    # Reset at beginning of a game
    # ------------------------------------------------------------

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
        self.puck_stationary_steps = 0

        self.paddle1_pos = np.array(
            [
                self._random_x(),
                self._random_blue_y()
            ],
            dtype=np.float32
        )

        self.paddle2_pos = np.array(
            [
                self._random_x(),
                self._random_red_y()
            ],
            dtype=np.float32
        )

        self.paddle1_vel = np.zeros(
            2,
            dtype=np.float32
        )

        self.paddle2_vel = np.zeros(
            2,
            dtype=np.float32
        )

        self.puck_pos = np.array(
            [
                self._random_x(),
                float(
                    self.np_random.uniform(
                        self.height
                        * RESET_PUCK_Y_MIN_FRACTION,
                        self.height
                        * RESET_PUCK_Y_MAX_FRACTION
                    )
                )
            ],
            dtype=np.float32
        )

        self.puck_vel = np.zeros(
            2,
            dtype=np.float32
        )

        return (
            self._get_observation(),
            info
        )


    # ------------------------------------------------------------
    # Reset after a goal
    # ------------------------------------------------------------

    def reset_after_goal(
        self,
        conceding_player
    ):

        self.paddle1_pos = np.array(
            [
                self._random_x(),
                self._random_blue_y()
            ],
            dtype=np.float32
        )

        self.paddle2_pos = np.array(
            [
                self._random_x(),
                self._random_red_y()
            ],
            dtype=np.float32
        )

        self.paddle1_vel = np.zeros(
            2,
            dtype=np.float32
        )

        self.paddle2_vel = np.zeros(
            2,
            dtype=np.float32
        )

        margin = (
            self.height
            * PUCK_HALF_MARGIN_FRACTION
        )

        if conceding_player == 1:

            puck_y = float(
                self.np_random.uniform(
                    self.height / 2 + margin,
                    self.height - margin
                )
            )

        elif conceding_player == 2:

            puck_y = float(
                self.np_random.uniform(
                    margin,
                    self.height / 2 - margin
                )
            )

        else:

            raise ValueError(
                "conceding_player must be 1 or 2"
            )

        self.puck_pos = np.array(
            [
                self._random_x(),
                puck_y
            ],
            dtype=np.float32
        )

        self.puck_vel = np.zeros(
            2,
            dtype=np.float32
        )

        # A goal reset must not count toward the 1.5-second
        # stuck-puck timer.
        self.puck_stationary_steps = 0

        return self._get_observation()


    # ------------------------------------------------------------
    # RED 100% chaser
    # ------------------------------------------------------------

    def _get_red_chaser_action(self):

        # RED chases once any part of the puck reaches RED's half.
        puck_on_red_side = (
            self.puck_pos[1]
            - self.puck_radius
            <= self.height / 2
        )

        if puck_on_red_side:

            target = self.puck_pos.copy()

        else:

            # Wait in the middle of RED's own half.
            target = np.array(
                [
                    self.width / 2,
                    self.height * 0.25
                ],
                dtype=np.float32
            )

        direction = (
            target
            - self.paddle2_pos
        )

        distance = float(
            np.linalg.norm(
                direction
            )
        )

        if distance <= 1e-8:

            return np.zeros(
                2,
                dtype=np.float32
            )

        return (
            direction / distance
        ).astype(np.float32)


    # ------------------------------------------------------------
    # Genuine single-event collision detection
    # ------------------------------------------------------------

    def _collision_and_detect(
        self,
        paddle_pos,
        paddle_vel
    ):

        difference = (
            self.puck_pos
            - paddle_pos
        )

        distance = float(
            np.linalg.norm(
                difference
            )
        )

        minimum_distance = (
            self.puck_radius
            + self.paddle_radius
        )

        if distance >= minimum_distance:

            return False

        if distance == 0:

            normal = np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        else:

            normal = (
                difference / distance
            )

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
            - paddle_normal_speed
        )

        # Only count a real incoming collision.
        real_hit = (
            relative_speed < 0
        )

        # Use the environment's normal physical collision.
        super()._handle_paddle_collision(
            paddle_pos,
            paddle_vel
        )

        return real_hit


    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def step(
        self,
        blue_action
    ):

        self.episode_steps += 1

        # BLUE action.
        blue_action = np.asarray(
            blue_action,
            dtype=np.float32
        )

        blue_action = np.clip(
            blue_action,
            -1.0,
            1.0
        )

        self.paddle1_vel = (
            blue_action
            * self.paddle_speed
        )

        # RED chaser action.
        red_action = (
            self._get_red_chaser_action()
        )

        self.paddle2_vel = (
            red_action
            * self.paddle_speed
            * CHASER_SPEED_FRACTION
        )

        # Move paddles.
        self.paddle1_pos += (
            self.paddle1_vel
        )

        self.paddle2_pos += (
            self.paddle2_vel
        )

        self._limit_paddles()

        # Start with NO step reward.
        reward = 0.0

        # Move puck.
        self.puck_pos += (
            self.puck_vel
        )

        self._handle_wall_collisions()

        # BLUE genuine hit.
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

        # RED physical collision only.
        player2_hit = (
            self._collision_and_detect(
                self.paddle2_pos,
                self.paddle2_vel
            )
        )

        # Re-check walls after paddle collisions.
        self._handle_wall_collisions()

        # --------------------------------------------------------
        # Stuck-puck rule
        # --------------------------------------------------------

        puck_speed = float(
            np.linalg.norm(
                self.puck_vel
            )
        )

        stall_speed_threshold = (
            self.max_puck_speed
            * PUCK_STALL_SPEED_FRACTION
        )

        if (
            puck_speed
            < stall_speed_threshold
        ):

            self.puck_stationary_steps += 1

        else:

            self.puck_stationary_steps = 0

        if (
            self.puck_stationary_steps
            >= PUCK_STALL_STEPS
        ):

            self.puck_pos = np.array(
                [
                    self.width / 2,
                    self.height / 2
                ],
                dtype=np.float32
            )

            self.puck_vel = np.zeros(
                2,
                dtype=np.float32
            )

            self.puck_stationary_steps = 0

        # --------------------------------------------------------
        # Goals
        # --------------------------------------------------------

        goal = self._check_goal()

        terminated = False
        truncated = False

        if goal == 1:

            reward += (
                self.goal_reward
            )

            observation = (
                self.reset_after_goal(
                    conceding_player=2
                )
            )

        elif goal == 2:

            reward += (
                self.concede_penalty
            )

            observation = (
                self.reset_after_goal(
                    conceding_player=1
                )
            )

        else:

            observation = (
                self._get_observation()
            )

        # Each training episode = 60 simulated seconds.
        if (
            self.training_mode
            and self.episode_steps
            >= TRAIN_EPISODE_MAX_STEPS
        ):

            truncated = True

        info = {
            "goal": goal,
            "player1_hit": player1_hit,
            "player2_hit": player2_hit,
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
    training_mode=True,
    render_mode=None,
    goal_reward=1.0,
    concede_penalty=-1.0,
    hit_reward=0.5
):

    return ThreeRewardFullChaserEnv(
        training_mode=training_mode,
        render_mode=render_mode,
        goal_reward=goal_reward,
        concede_penalty=concede_penalty,
        hit_reward=hit_reward
    )


# ================================================================
# EVALUATE ONE AGENT
# ================================================================

def evaluate_agent(
    model,
    evaluation_seed_start,
    goal_reward,
    concede_penalty,
    hit_reward
):

    env = make_env(
        training_mode=False,
        render_mode=None,
        goal_reward=goal_reward,
        concede_penalty=concede_penalty,
        hit_reward=hit_reward
    )

    blue_scores = []
    red_scores = []

    for game_number in range(
        EVALUATION_GAMES
    ):

        observation, info = env.reset(
            seed=(
                evaluation_seed_start
                + game_number
            )
        )

        blue_goals = 0
        red_goals = 0

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

            # Evaluation env is not time-truncated internally,
            # but keep this safe for any future environment change.
            if terminated or truncated:

                observation, info = env.reset()

        blue_scores.append(
            blue_goals
        )

        red_scores.append(
            red_goals
        )

    env.close()

    average_blue = float(
        np.mean(
            blue_scores
        )
    )

    average_red = float(
        np.mean(
            red_scores
        )
    )

    return {
        "blue_goals_per_game":
            average_blue,

        "red_goals_per_game":
            average_red,

        "goal_difference":
            average_blue
            - average_red,
    }


# ================================================================
# OPTUNA OBJECTIVE
# ================================================================

ALL_RESULTS = []


def objective(
    trial
):

    # Optuna tunes ALL THREE rewards.
    goal_reward = trial.suggest_float(
        "goal_reward",
        GOAL_REWARD_MIN,
        GOAL_REWARD_MAX
    )

    concede_penalty = trial.suggest_float(
        "concede_penalty",
        CONCEDE_PENALTY_MIN,
        CONCEDE_PENALTY_MAX
    )

    hit_reward = trial.suggest_float(
        "hit_reward",
        HIT_REWARD_MIN,
        HIT_REWARD_MAX
    )

    seed_goal_differences = []

    for seed in TRAINING_SEEDS:

        # --------------------------------------------------------
        # Fresh environment and fresh PPO agent
        # --------------------------------------------------------

        env = make_env(
            training_mode=True,
            render_mode=None,
            goal_reward=goal_reward,
            concede_penalty=concede_penalty,
            hit_reward=hit_reward
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

        # Silent 500k training.
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            reset_num_timesteps=True,
            progress_bar=False
        )

        # --------------------------------------------------------
        # Save model
        # --------------------------------------------------------

        trial_folder = os.path.join(
            MODEL_FOLDER,
            f"trial_{trial.number + 1}"
        )

        os.makedirs(
            trial_folder,
            exist_ok=True
        )

        model_path = os.path.join(
            trial_folder,
            f"seed_{seed}"
        )

        model.save(
            model_path
        )

        env.close()

        # --------------------------------------------------------
        # 100 x 60-second evaluation games
        # --------------------------------------------------------

        metrics = evaluate_agent(
            model=model,
            evaluation_seed_start=(
                100_000
                + trial.number * 10_000
                + seed * 1_000
            ),
            goal_reward=goal_reward,
            concede_penalty=concede_penalty,
            hit_reward=hit_reward
        )

        goal_difference = float(
            metrics[
                "goal_difference"
            ]
        )

        seed_goal_differences.append(
            goal_difference
        )

        row = {
            "trial":
                trial.number + 1,

            "seed":
                seed,

            "goal_reward":
                goal_reward,

            "concede_penalty":
                concede_penalty,

            "hit_reward":
                hit_reward,

            "timesteps":
                TRAIN_TIMESTEPS,

            **metrics
        }

        ALL_RESULTS.append(
            row
        )

        pd.DataFrame(
            ALL_RESULTS
        ).to_csv(
            CSV_PATH,
            index=False
        )

        # Requested output: evaluation results only.
        print(
            f"Trial {trial.number + 1} "
            f"Seed {seed} | "
            f"BLUE {metrics['blue_goals_per_game']:.3f} | "
            f"RED {metrics['red_goals_per_game']:.3f} | "
            f"DIFF {metrics['goal_difference']:+.3f}"
        )

    # Optuna maximises MEDIAN GOAL DIFFERENCE across the three seeds.
    median_goal_difference = float(
        np.median(
            seed_goal_differences
        )
    )

    trial.set_user_attr(
        "seed_goal_differences",
        seed_goal_differences
    )

    trial.set_user_attr(
        "mean_goal_difference",
        float(
            np.mean(
                seed_goal_differences
            )
        )
    )

    trial.set_user_attr(
        "median_goal_difference",
        median_goal_difference
    )

    return median_goal_difference


# ================================================================
# RUN OPTUNA
# ================================================================

def run_optuna():

    # Hide Optuna's normal trial logging.
    optuna.logging.set_verbosity(
        optuna.logging.CRITICAL
    )

    # Validate environment once before the long run.
    check_environment = make_env(
        training_mode=True
    )

    check_env(
        check_environment,
        warn=False
    )

    check_environment.close()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_STORAGE,
        direction="maximize",
        load_if_exists=True
    )

    remaining_trials = max(
        0,
        N_TRIALS
        - len(study.trials)
    )

    if remaining_trials > 0:

        study.optimize(
            objective,
            n_trials=remaining_trials
        )

    return study


# ================================================================
# FINAL SUMMARY
# ================================================================

def print_final_summary(
    study
):

    print()
    print("=" * 72)
    print("FINAL OPTUNA RESULTS")
    print("=" * 72)

    completed = [
        trial
        for trial in study.trials
        if trial.value is not None
    ]

    for trial in completed:

        print(
            f"Trial {trial.number + 1} | "
            f"Goal {trial.params['goal_reward']:+.4f} | "
            f"Concede {trial.params['concede_penalty']:+.4f} | "
            f"Hit {trial.params['hit_reward']:+.4f} | "
            f"Median DIFF {trial.value:+.3f}"
        )

    if completed:

        best = study.best_trial

        print()
        print("BEST TRIAL")
        print(
            f"Trial {best.number + 1}"
        )
        print(
            f"Goal reward:       "
            f"{best.params['goal_reward']:+.4f}"
        )
        print(
            f"Concede penalty:   "
            f"{best.params['concede_penalty']:+.4f}"
        )
        print(
            f"Hit reward:        "
            f"{best.params['hit_reward']:+.4f}"
        )
        print(
            f"Median goal diff:  "
            f"{best.value:+.3f}"
        )

#%%

# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    study = run_optuna()

    print_final_summary(
        study
    )
#%%
# ================================================================
# CELL 2 - WATCH TRAINED AGENT VS 100% CHASER
# ================================================================

import os
import pygame
import optuna

from stable_baselines3 import PPO


# ================================================================
# CHOOSE AGENT
# ================================================================

TRIAL_TO_WATCH = 1
SEED_TO_WATCH = 2

WATCH_SECONDS = 60
WATCH_FPS = 60

TOTAL_STEPS = (
    WATCH_SECONDS
    * WATCH_FPS
)


# ================================================================
# MODEL PATH
# ================================================================

MODEL_PATH = os.path.join(
    "3rewards_full_chaser",
    "models",
    f"trial_{TRIAL_TO_WATCH}",
    f"seed_{SEED_TO_WATCH}.zip"
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Could not find model:\n{MODEL_PATH}"
    )


# ================================================================
# LOAD OPTUNA TRIAL
# ================================================================

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=OPTUNA_STORAGE
)


trial = study.trials[
    TRIAL_TO_WATCH - 1
]


goal_reward = trial.params[
    "goal_reward"
]

concede_penalty = trial.params[
    "concede_penalty"
]

hit_reward = trial.params[
    "hit_reward"
]


# ================================================================
# LOAD MODEL
# ================================================================

model = PPO.load(
    MODEL_PATH
)


# ================================================================
# CREATE VISIBLE ENVIRONMENT
# ================================================================

pygame.init()


env = make_env(
    training_mode=False,
    render_mode="human",
    goal_reward=goal_reward,
    concede_penalty=concede_penalty,
    hit_reward=hit_reward
)


observation, info = env.reset(
    seed=42
)


pygame.display.set_caption(
    f"Trial {TRIAL_TO_WATCH} - Seed {SEED_TO_WATCH}"
)


# ================================================================
# SCORE
# ================================================================

blue_goals = 0
red_goals = 0

running = True


# ================================================================
# GAME LOOP
# ================================================================

for step in range(
    TOTAL_STEPS
):

    # ------------------------------------------------------------
    # SERVICE WINDOWS / PYGAME EVENTS
    # ------------------------------------------------------------

    pygame.event.pump()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        elif (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
        ):

            running = False


    if not running:

        break


    # ------------------------------------------------------------
    # AI ACTION
    # ------------------------------------------------------------

    action, _ = model.predict(
        observation,
        deterministic=True
    )


    # Give Windows another opportunity to process the window.
    pygame.event.pump()


    # ------------------------------------------------------------
    # ENVIRONMENT STEP
    # ------------------------------------------------------------

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(
        action
    )


    # ------------------------------------------------------------
    # SCORE
    # ------------------------------------------------------------

    if info["goal"] == 1:

        blue_goals += 1


    elif info["goal"] == 2:

        red_goals += 1


    # Important:
    # NO clock.tick() here.
    #
    # env.render() already limits the Pygame window to 60 FPS.


# ================================================================
# CLEAN CLOSE
# ================================================================

env.close()


# Process final window messages before quitting.
pygame.event.pump()

pygame.quit()


# ================================================================
# RESULT
# ================================================================

print()
print("=" * 60)

print(
    f"TRIAL {TRIAL_TO_WATCH} "
    f"| SEED {SEED_TO_WATCH}"
)

print("=" * 60)

print(
    f"BLUE {blue_goals} - {red_goals} RED"
)