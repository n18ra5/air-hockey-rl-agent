# -*- coding: utf-8 -*-
"""
BALANCED AIR-HOCKEY - SIMPLE REWARD OPTUNA TEST
================================================

BLUE:
    PPO agent continuing from Stage 1 Trial 4 Seed 1.

RED:
    50%-speed chaser opponent.

REWARDS
=======
Blue scores:                         +1.0   FIXED
Blue concedes:                       -1.0   FIXED
Blue makes a real puck hit:          Optuna-tuned existing reward
Extra real-hit contact bonus:        +0.01  FIXED (NEW)
Gets closer to puck on BLUE half:    Optuna tuned
Does NOT get closer on BLUE half:    Optuna tuned
Slow movement:                       Optuna tuned

NEW CHANGES IN THIS VERSION
===========================
1. Every real BLUE-puck collision gets an additional fixed +0.01 reward.
2. Puck-stall detection now uses actual frame-to-frame puck movement.
   Movement <= 0.5 table units per step counts as effectively stationary.
   The reset still requires 1.5 continuous seconds.

No other reward-system changes have been added.
"""

import os
import numpy as np
import pandas as pd
import optuna
import pygame

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from air_hockey_env import AirHockeyEnv


# ================================================================
# QUICK TEST / FULL TRAINING
# ================================================================

QUICK_TEST = False

if QUICK_TEST:
    TRAIN_TIMESTEPS = 10_000
    EVALUATION_GAMES = 5
    EVALUATION_SECONDS = 10
else:
    TRAIN_TIMESTEPS = 200_000
    EVALUATION_GAMES = 100
    EVALUATION_SECONDS = 60

TRAINING_SEEDS = [1, 2, 3]
SIMULATION_FPS = 60
TRAIN_MATCH_SECONDS = 60

TRAIN_EPISODE_MAX_STEPS = TRAIN_MATCH_SECONDS * SIMULATION_FPS
EVALUATION_STEPS_PER_GAME = EVALUATION_SECONDS * SIMULATION_FPS


# ================================================================
# FIXED REWARDS
# ================================================================

GOAL_REWARD = 1.0
CONCEDE_PENALTY = -1.0

DEFAULT_HIT_REWARD = 0.5
DEFAULT_APPROACH_REWARD = 0.01
DEFAULT_NO_APPROACH_PENALTY = -0.01
DEFAULT_SLOW_MOVEMENT_PENALTY = -0.01

# NEW REQUESTED REWARD:
# Extra fixed reward every time BLUE makes a real puck collision.
EXTRA_REAL_HIT_BONUS = 0.01


# ================================================================
# OPTUNA SEARCH RANGES
# ================================================================

HIT_REWARD_MIN = 0.10
HIT_REWARD_MAX = 0.80

APPROACH_REWARD_MIN = 0.0005
APPROACH_REWARD_MAX = 0.0150

NO_APPROACH_PENALTY_MIN = -0.0150
NO_APPROACH_PENALTY_MAX = -0.0005

SLOW_MOVEMENT_PENALTY_MIN = -0.0200
SLOW_MOVEMENT_PENALTY_MAX = -0.0010

SLOW_SPEED_FRACTION = 0.10


# ================================================================
# RANDOMISED START / RESTART POSITIONS
# ================================================================

RANDOM_X_MIN_FRACTION = 0.20
RANDOM_X_MAX_FRACTION = 0.80

BLUE_Y_MIN_FRACTION = 0.60
BLUE_Y_MAX_FRACTION = 0.90

RED_Y_MIN_FRACTION = 0.10
RED_Y_MAX_FRACTION = 0.40

RESET_PUCK_Y_MIN_FRACTION = 0.35
RESET_PUCK_Y_MAX_FRACTION = 0.65

PUCK_HALF_MARGIN_FRACTION = 0.08

# Puck must remain effectively stuck for 1.5 s.
PUCK_STALL_SECONDS = 1.5
PUCK_STALL_STEPS = int(PUCK_STALL_SECONDS * SIMULATION_FPS)

# NEW THRESHOLD CHOICE:
# Tiny corner vibration of up to 0.5 table units in a frame is ignored.
PUCK_STALL_MOVEMENT_THRESHOLD = 0.5


# ================================================================
# OUTPUT
# ================================================================
# New folder/study prevents the old completed Optuna study from causing
# this new run to immediately stop with zero remaining trials.

if QUICK_TEST:
    OUTPUT_FOLDER = "balanced_ai_simple_rewards_randomised_hitbonus_stallfix_test"
else:
    OUTPUT_FOLDER = "balanced_ai_simple_rewards_randomised_hitbonus_stallfix"

MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_FOLDER, "all_trial_results.csv")
OPTUNA_DB_PATH = os.path.join(
    OUTPUT_FOLDER,
    "balanced_simple_rewards_randomised_hitbonus_stallfix_optuna.db"
)
OPTUNA_STORAGE = "sqlite:///" + os.path.abspath(OPTUNA_DB_PATH)

N_TRIALS = 5


# ================================================================
# STAGE 2 STARTING MODEL
# ================================================================

STAGE1_MODEL_FOLDER = os.path.join(
    "balanced_ai_optuna",
    "models",
    "trial_4"
)

STAGE1_STARTING_MODEL = os.path.join(
    STAGE1_MODEL_FOLDER,
    "balanced_seed_1.zip"
)


# ================================================================
# CHASER SETTINGS
# ================================================================

CHASER_SPEED_FRACTION = 0.50


# ================================================================
# TRAINING OUTPUT CALLBACK
# ================================================================

class TrainingProgressCallback(BaseCallback):

    def __init__(self, print_every=25_000, rolling_episodes=100):
        super().__init__(verbose=0)
        self.print_every = int(print_every)
        self.next_print_step = int(print_every)
        self.rolling_episodes = int(rolling_episodes)
        self.current_episode_reward = 0.0
        self.completed_episode_rewards = []

    def _on_step(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None:
            self.current_episode_reward += float(
                np.asarray(rewards).reshape(-1)[0]
            )

        if dones is not None:
            done = bool(np.asarray(dones).reshape(-1)[0])
            if done:
                self.completed_episode_rewards.append(
                    self.current_episode_reward
                )
                if len(self.completed_episode_rewards) > self.rolling_episodes:
                    self.completed_episode_rewards = self.completed_episode_rewards[
                        -self.rolling_episodes:
                    ]
                self.current_episode_reward = 0.0

        while self.num_timesteps >= self.next_print_step:
            if self.completed_episode_rewards:
                average_reward = float(np.mean(self.completed_episode_rewards))
                reward_text = f"{average_reward:.3f}"
            else:
                reward_text = "no completed games yet"

            explained_variance = self.model.logger.name_to_value.get(
                "train/explained_variance",
                np.nan
            )

            try:
                explained_variance = float(explained_variance)
            except (TypeError, ValueError):
                explained_variance = np.nan

            explained_text = (
                f"{explained_variance:.3f}"
                if np.isfinite(explained_variance)
                else "N/A"
            )

            print(
                f"{self.next_print_step:>7,} steps | "
                f"average reward/game: {reward_text} | "
                f"explained variance: {explained_text}"
            )

            self.next_print_step += self.print_every

        return True


# ================================================================
# BALANCED TRAINING ENVIRONMENT
# ================================================================

class BalancedTrainingEnv(AirHockeyEnv):

    def __init__(
        self,
        render_mode=None,
        training_mode=True,
        hit_reward=DEFAULT_HIT_REWARD,
        approach_reward=DEFAULT_APPROACH_REWARD,
        no_approach_penalty=DEFAULT_NO_APPROACH_PENALTY,
        slow_movement_penalty=DEFAULT_SLOW_MOVEMENT_PENALTY
    ):
        super().__init__(render_mode=render_mode)

        self.training_mode = training_mode
        self.hit_reward = float(hit_reward)
        self.approach_reward = float(approach_reward)
        self.no_approach_penalty = float(no_approach_penalty)
        self.slow_movement_penalty = float(slow_movement_penalty)

        # PPO controls BLUE only.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.episode_steps = 0
        self.puck_stationary_steps = 0
        self.previous_puck_stall_pos = None

    # ------------------------------------------------------------
    # Random position helpers
    # ------------------------------------------------------------

    def _random_x(self):
        return float(self.np_random.uniform(
            self.width * RANDOM_X_MIN_FRACTION,
            self.width * RANDOM_X_MAX_FRACTION
        ))

    def _random_blue_y(self):
        return float(self.np_random.uniform(
            self.height * BLUE_Y_MIN_FRACTION,
            self.height * BLUE_Y_MAX_FRACTION
        ))

    def _random_red_y(self):
        return float(self.np_random.uniform(
            self.height * RED_Y_MIN_FRACTION,
            self.height * RED_Y_MAX_FRACTION
        ))

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        self.episode_steps = 0
        self.puck_stationary_steps = 0

        self.paddle1_pos = np.array(
            [self._random_x(), self._random_blue_y()],
            dtype=np.float32
        )
        self.paddle2_pos = np.array(
            [self._random_x(), self._random_red_y()],
            dtype=np.float32
        )

        self.paddle1_vel = np.zeros(2, dtype=np.float32)
        self.paddle2_vel = np.zeros(2, dtype=np.float32)

        self.puck_pos = np.array(
            [
                self._random_x(),
                float(self.np_random.uniform(
                    self.height * RESET_PUCK_Y_MIN_FRACTION,
                    self.height * RESET_PUCK_Y_MAX_FRACTION
                ))
            ],
            dtype=np.float32
        )
        self.puck_vel = np.zeros(2, dtype=np.float32)
        self.previous_puck_stall_pos = self.puck_pos.copy()

        return self._get_observation(), info

    # ------------------------------------------------------------
    # Reset after goal
    # ------------------------------------------------------------

    def reset_after_goal(self, conceding_player):
        self.paddle1_pos = np.array(
            [self._random_x(), self._random_blue_y()],
            dtype=np.float32
        )
        self.paddle2_pos = np.array(
            [self._random_x(), self._random_red_y()],
            dtype=np.float32
        )

        self.paddle1_vel = np.zeros(2, dtype=np.float32)
        self.paddle2_vel = np.zeros(2, dtype=np.float32)

        margin = self.height * PUCK_HALF_MARGIN_FRACTION

        if conceding_player == 1:
            puck_y = float(self.np_random.uniform(
                self.height / 2 + margin,
                self.height - margin
            ))
        elif conceding_player == 2:
            puck_y = float(self.np_random.uniform(
                margin,
                self.height / 2 - margin
            ))
        else:
            raise ValueError("conceding_player must be 1 or 2")

        self.puck_pos = np.array(
            [self._random_x(), puck_y],
            dtype=np.float32
        )
        self.puck_vel = np.zeros(2, dtype=np.float32)
        self.puck_stationary_steps = 0
        self.previous_puck_stall_pos = self.puck_pos.copy()

        return self._get_observation()

    # ------------------------------------------------------------
    # RED chaser
    # ------------------------------------------------------------

    def _get_red_chaser_action(self):
        puck_on_red_side = (
            self.puck_pos[1] - self.puck_radius <= self.height / 2
        )

        if puck_on_red_side:
            target = self.puck_pos.copy()
        else:
            target = np.array(
                [self.width / 2, self.height * 0.25],
                dtype=np.float32
            )

        direction = target - self.paddle2_pos
        distance = float(np.linalg.norm(direction))

        if distance <= 1e-8:
            return np.zeros(2, dtype=np.float32)

        return (direction / distance).astype(np.float32)

    # ------------------------------------------------------------
    # Real hit detection + original collision physics
    # ------------------------------------------------------------

    def _collision_and_detect(self, paddle_pos, paddle_vel):
        difference = self.puck_pos - paddle_pos
        distance = float(np.linalg.norm(difference))
        minimum_distance = self.puck_radius + self.paddle_radius

        if distance >= minimum_distance:
            return False

        if distance == 0:
            normal = np.array([1.0, 0.0], dtype=np.float32)
        else:
            normal = difference / distance

        puck_normal_speed = float(np.dot(self.puck_vel, normal))
        paddle_normal_speed = float(np.dot(paddle_vel, normal))
        relative_speed = puck_normal_speed - paddle_normal_speed
        real_hit = relative_speed < 0

        super()._handle_paddle_collision(paddle_pos, paddle_vel)
        return real_hit

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def step(self, blue_action):
        self.episode_steps += 1

        blue_action = np.asarray(blue_action, dtype=np.float32)
        blue_action = np.clip(blue_action, -1.0, 1.0)

        new_blue_velocity = blue_action * self.paddle_speed

        old_distance_to_puck = float(np.linalg.norm(
            self.puck_pos - self.paddle1_pos
        ))

        red_action = self._get_red_chaser_action()
        new_red_velocity = (
            red_action * self.paddle_speed * CHASER_SPEED_FRACTION
        )

        reward = 0.0

        # Slow-movement penalty.
        blue_speed = float(np.linalg.norm(new_blue_velocity))
        slow_speed_threshold = self.paddle_speed * SLOW_SPEED_FRACTION

        if blue_speed < slow_speed_threshold:
            reward += self.slow_movement_penalty

        # Apply paddle movement.
        self.paddle1_vel = new_blue_velocity
        self.paddle2_vel = new_red_velocity

        self.paddle1_pos += self.paddle1_vel
        self.paddle2_pos += self.paddle2_vel
        self._limit_paddles()

        # Approach / no-approach reward only on BLUE half.
        puck_on_blue_half = self.puck_pos[1] >= self.height / 2

        new_distance_to_puck = float(np.linalg.norm(
            self.puck_pos - self.paddle1_pos
        ))

        if puck_on_blue_half:
            if new_distance_to_puck < old_distance_to_puck:
                reward += self.approach_reward
            else:
                reward += self.no_approach_penalty

        # Move puck and handle walls.
        self.puck_pos += self.puck_vel
        self._handle_wall_collisions()

        # BLUE real hit.
        player1_hit = self._collision_and_detect(
            self.paddle1_pos,
            self.paddle1_vel
        )

        if player1_hit:
            # Existing Optuna-tuned hit reward.
            reward += self.hit_reward

            # NEW: additional fixed reward for actually making contact.
            reward += EXTRA_REAL_HIT_BONUS

        # RED physical hit only; no BLUE reward/penalty.
        player2_hit = self._collision_and_detect(
            self.paddle2_pos,
            self.paddle2_vel
        )

        # --------------------------------------------------------
        # NEW PUCK-STALL DETECTION
        # --------------------------------------------------------
        # Compare actual puck displacement from one completed step to
        # the next. Tiny corner vibration <= threshold is ignored.

        if self.previous_puck_stall_pos is None:
            self.previous_puck_stall_pos = self.puck_pos.copy()

        puck_movement = float(np.linalg.norm(
            self.puck_pos - self.previous_puck_stall_pos
        ))

        if puck_movement <= PUCK_STALL_MOVEMENT_THRESHOLD:
            self.puck_stationary_steps += 1
        else:
            self.puck_stationary_steps = 0

        self.previous_puck_stall_pos = self.puck_pos.copy()

        if self.puck_stationary_steps >= PUCK_STALL_STEPS:
            self.puck_pos = np.array(
                [self.width / 2, self.height / 2],
                dtype=np.float32
            )
            self.puck_vel = np.zeros(2, dtype=np.float32)
            self.puck_stationary_steps = 0
            self.previous_puck_stall_pos = self.puck_pos.copy()

        # Goals.
        goal = self._check_goal()
        terminated = False
        truncated = False

        if goal == 1:
            reward += GOAL_REWARD
            observation = self.reset_after_goal(conceding_player=2)
        elif goal == 2:
            reward += CONCEDE_PENALTY
            observation = self.reset_after_goal(conceding_player=1)
        else:
            observation = self._get_observation()

        if (
            self.training_mode
            and self.episode_steps >= TRAIN_EPISODE_MAX_STEPS
        ):
            truncated = True

        info = {
            "goal": goal,
            "player1_hit": player1_hit,
            "player2_hit": player2_hit,
            "puck_movement": puck_movement,
            "puck_stationary_steps": self.puck_stationary_steps,
        }

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info


# ================================================================
# MAKE ENVIRONMENT
# ================================================================

def make_env(
    training_mode=True,
    render_mode=None,
    hit_reward=DEFAULT_HIT_REWARD,
    approach_reward=DEFAULT_APPROACH_REWARD,
    no_approach_penalty=DEFAULT_NO_APPROACH_PENALTY,
    slow_movement_penalty=DEFAULT_SLOW_MOVEMENT_PENALTY
):
    return BalancedTrainingEnv(
        training_mode=training_mode,
        render_mode=render_mode,
        hit_reward=hit_reward,
        approach_reward=approach_reward,
        no_approach_penalty=no_approach_penalty,
        slow_movement_penalty=slow_movement_penalty
    )


# ================================================================
# EVALUATE ONE AGENT
# ================================================================

def evaluate_agent(
    model,
    evaluation_seed_start,
    hit_reward,
    approach_reward,
    no_approach_penalty,
    slow_movement_penalty
):
    env = make_env(
        training_mode=False,
        hit_reward=hit_reward,
        approach_reward=approach_reward,
        no_approach_penalty=no_approach_penalty,
        slow_movement_penalty=slow_movement_penalty
    )

    blue_scores = []
    red_scores = []
    rewards_per_game = []

    for game_number in range(EVALUATION_GAMES):
        observation, info = env.reset(
            seed=evaluation_seed_start + game_number
        )

        blue_goals = 0
        red_goals = 0
        total_game_reward = 0.0

        for _ in range(EVALUATION_STEPS_PER_GAME):
            action, _ = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            total_game_reward += float(reward)

            if info["goal"] == 1:
                blue_goals += 1
            elif info["goal"] == 2:
                red_goals += 1

            if terminated or truncated:
                observation, info = env.reset()

        blue_scores.append(blue_goals)
        red_scores.append(red_goals)
        rewards_per_game.append(total_game_reward)

    env.close()

    mean_blue = float(np.mean(blue_scores))
    mean_red = float(np.mean(red_scores))

    return {
        "blue_goals_per_game": mean_blue,
        "red_goals_per_game": mean_red,
        "goal_difference": mean_blue - mean_red,
        "average_reward_per_game": float(np.mean(rewards_per_game))
    }


# ================================================================
# OPTUNA OBJECTIVE
# ================================================================

ALL_RESULTS = []


def objective(trial):
    hit_reward = trial.suggest_float(
        "hit_reward",
        HIT_REWARD_MIN,
        HIT_REWARD_MAX
    )

    approach_reward = trial.suggest_float(
        "approach_reward",
        APPROACH_REWARD_MIN,
        APPROACH_REWARD_MAX
    )

    no_approach_penalty = trial.suggest_float(
        "no_approach_penalty",
        NO_APPROACH_PENALTY_MIN,
        NO_APPROACH_PENALTY_MAX
    )

    slow_movement_penalty = trial.suggest_float(
        "slow_movement_penalty",
        SLOW_MOVEMENT_PENALTY_MIN,
        SLOW_MOVEMENT_PENALTY_MAX
    )

    seed_scores = []

    for seed in TRAINING_SEEDS:
        print()
        print(f"Trial {trial.number + 1} | Seed {seed}")

        env = make_env(
            training_mode=True,
            hit_reward=hit_reward,
            approach_reward=approach_reward,
            no_approach_penalty=no_approach_penalty,
            slow_movement_penalty=slow_movement_penalty
        )

        env.reset(seed=seed)

        if not os.path.exists(STAGE1_STARTING_MODEL):
            raise FileNotFoundError(
                "Could not find Stage 1 Trial 4 Seed 1 model: "
                f"{STAGE1_STARTING_MODEL}"
            )

        model = PPO.load(
            STAGE1_STARTING_MODEL,
            env=env,
            device="auto"
        )
        model.set_random_seed(seed)
        model.verbose = 0

        progress_callback = TrainingProgressCallback(
            print_every=25_000,
            rolling_episodes=100
        )

        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            callback=progress_callback,
            reset_num_timesteps=True
        )

        trial_folder = os.path.join(
            MODEL_FOLDER,
            f"trial_{trial.number + 1}"
        )
        os.makedirs(trial_folder, exist_ok=True)

        model_path = os.path.join(
            trial_folder,
            f"balanced_seed_{seed}"
        )
        model.save(model_path)
        env.close()

        metrics = evaluate_agent(
            model=model,
            evaluation_seed_start=(
                30_000
                + trial.number * 10_000
                + seed * 1_000
            ),
            hit_reward=hit_reward,
            approach_reward=approach_reward,
            no_approach_penalty=no_approach_penalty,
            slow_movement_penalty=slow_movement_penalty
        )

        seed_scores.append(metrics["blue_goals_per_game"])

        row = {
            "trial": trial.number + 1,
            "seed": seed,
            "hit_reward": hit_reward,
            "extra_real_hit_bonus": EXTRA_REAL_HIT_BONUS,
            "approach_reward": approach_reward,
            "no_approach_penalty": no_approach_penalty,
            "slow_movement_penalty": slow_movement_penalty,
            "puck_stall_movement_threshold": PUCK_STALL_MOVEMENT_THRESHOLD,
            "timesteps": TRAIN_TIMESTEPS,
            **metrics
        }

        ALL_RESULTS.append(row)
        pd.DataFrame(ALL_RESULTS).to_csv(CSV_PATH, index=False)

        print()
        print(f"After {EVALUATION_GAMES} evaluation games:")
        print(
            f"Average BLUE goals/game: "
            f"{metrics['blue_goals_per_game']:.3f}"
        )
        print(
            f"Average RED goals/game:  "
            f"{metrics['red_goals_per_game']:.3f}"
        )
        print(
            f"Average reward/game:     "
            f"{metrics['average_reward_per_game']:.3f}"
        )

    median_score = float(np.median(seed_scores))

    trial.set_user_attr("seed_scores", seed_scores)
    trial.set_user_attr("mean_blue_goals", float(np.mean(seed_scores)))
    trial.set_user_attr("variation", float(np.std(seed_scores, ddof=0)))

    return median_score


# ================================================================
# RUN OPTUNA STUDY
# ================================================================

def run_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    check_environment = make_env(training_mode=True)
    check_env(check_environment, warn=False)
    check_environment.close()

    study = optuna.create_study(
        study_name="balanced_air_hockey_simple_rewards_randomised_hitbonus_stallfix",
        storage=OPTUNA_STORAGE,
        direction="maximize",
        load_if_exists=True
    )

    remaining_trials = max(0, N_TRIALS - len(study.trials))

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)

    return study


# ================================================================
# WATCH A TRAINED MODEL
# ================================================================

def watch_model(trial_number, seed=1, seconds=60):
    study = optuna.load_study(
        study_name="balanced_air_hockey_simple_rewards_randomised_hitbonus_stallfix",
        storage=OPTUNA_STORAGE
    )

    matching_trials = [
        t for t in study.trials
        if t.number + 1 == trial_number
    ]

    if not matching_trials:
        raise ValueError(f"Trial {trial_number} does not exist.")

    selected_trial = matching_trials[0]

    model_path = os.path.join(
        MODEL_FOLDER,
        f"trial_{trial_number}",
        f"balanced_seed_{seed}.zip"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find {model_path}.")

    model = PPO.load(model_path)

    env = make_env(
        training_mode=False,
        render_mode="human",
        hit_reward=selected_trial.params["hit_reward"],
        approach_reward=selected_trial.params["approach_reward"],
        no_approach_penalty=selected_trial.params["no_approach_penalty"],
        slow_movement_penalty=selected_trial.params["slow_movement_penalty"]
    )

    observation, info = env.reset(seed=50_000 + seed)
    clock = pygame.time.Clock()

    blue_goals = 0
    red_goals = 0
    running = True
    steps = 0
    max_steps = int(seconds * SIMULATION_FPS)

    while running and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        if info["goal"] == 1:
            blue_goals += 1
        elif info["goal"] == 2:
            red_goals += 1

        steps += 1
        clock.tick(SIMULATION_FPS)

    env.close()

    print()
    print(f"Trial {trial_number}, seed {seed}")
    print(f"Final score after {steps / SIMULATION_FPS:.1f}s:")
    print(f"BLUE {blue_goals} - {red_goals} RED")


#%%
# ================================================================
# CELL 1 - RUN STAGE 2 TRAINING
# ================================================================

study = run_optuna()


#%%
# ================================================================
# CELL 2 - WATCH A TRAINED STAGE 2 MODEL
# ================================================================

TRIAL_TO_WATCH = 3
SEED_TO_WATCH = 1
WATCH_SECONDS = 60
#1,3 2,1 3,1
watch_model(
    trial_number=TRIAL_TO_WATCH,
    seed=SEED_TO_WATCH,
    seconds=WATCH_SECONDS
)
