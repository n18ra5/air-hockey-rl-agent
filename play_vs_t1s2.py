# -*- coding: utf-8 -*-
"""
PLAY AGAINST T1S2
=================

BLUE:
    You control the paddle with your mouse / touchpad.

RED:
    Frozen Trial 1 Seed 2 agent from:
        3rewards_full_chaser/models/trial_1/seed_2.zip

IMPORTANT:
    T1S2 was originally trained as the BOTTOM / BLUE player.

    To let it play as the TOP / RED player here, its observation is
    rotated by 180 degrees before being passed to the model.

    Its chosen action is then rotated back into the real table.

Controls:
    Mouse / touchpad = move BLUE paddle
    ESC              = quit
"""

import os
import numpy as np
import pygame

from stable_baselines3 import PPO

from air_hockey_env import AirHockeyEnv


# ================================================================
# SETTINGS
# ================================================================

MODEL_PATH = os.path.join(
    "3rewards_full_chaser",
    "models",
    "trial_1",
    "seed_2.zip"
)


# ================================================================
# CHECK MODEL
# ================================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Could not find T1S2 model:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )


# ================================================================
# LOAD FROZEN T1S2
# ================================================================

T1S2 = PPO.load(
    MODEL_PATH
)


# ================================================================
# BLUE HUMAN CONTROL
# ================================================================

def get_blue_action(env):
    """
    Convert mouse/touchpad position into BLUE's movement action.
    """

    mouse_x, mouse_y = pygame.mouse.get_pos()

    target = np.array(
        [
            mouse_x - env.border,
            mouse_y - env.border
        ],
        dtype=np.float32
    )

    r = env.paddle_radius

    target[0] = np.clip(
        target[0],
        r,
        env.width - r
    )

    target[1] = np.clip(
        target[1],
        env.height / 2 + r,
        env.height - r
    )

    displacement = (
        target
        - env.paddle1_pos
    )

    blue_action = np.clip(
        displacement / env.paddle_speed,
        -1.0,
        1.0
    )

    return blue_action.astype(
        np.float32
    )


# ================================================================
# BUILD T1S2'S RED-SIDE OBSERVATION
# ================================================================

def get_t1s2_red_observation(env):
    """
    Rotate the whole table by 180 degrees so the real RED player
    appears to T1S2 as its familiar BLUE/bottom-side player.
    """

    return np.array(
        [
            # Puck
            env.width - env.puck_pos[0],
            env.height - env.puck_pos[1],
            -env.puck_vel[0],
            -env.puck_vel[1],

            # Real RED becomes T1S2's own paddle
            env.width - env.paddle2_pos[0],
            env.height - env.paddle2_pos[1],
            -env.paddle2_vel[0],
            -env.paddle2_vel[1],

            # Real BLUE becomes T1S2's opponent
            env.width - env.paddle1_pos[0],
            env.height - env.paddle1_pos[1],
            -env.paddle1_vel[0],
            -env.paddle1_vel[1],
        ],
        dtype=np.float32
    )


# ================================================================
# GET T1S2 ACTION
# ================================================================

def get_t1s2_action(env):

    red_observation = (
        get_t1s2_red_observation(env)
    )

    model_action, _ = T1S2.predict(
        red_observation,
        deterministic=True
    )

    model_action = np.asarray(
        model_action,
        dtype=np.float32
    )

    model_action = np.clip(
        model_action,
        -1.0,
        1.0
    )

    # Rotate the model's action back into real table coordinates.
    red_action = np.array(
        [
            -model_action[0],
            -model_action[1]
        ],
        dtype=np.float32
    )

    return red_action


# ================================================================
# MAIN GAME
# ================================================================

def main():

    pygame.init()

    env = AirHockeyEnv(
        render_mode="human"
    )

    observation, info = env.reset()

    pygame.display.set_caption(
        "Air Hockey - YOU vs T1S2"
    )

    print()
    print("=" * 60)
    print("AIR HOCKEY - YOU vs T1S2")
    print("=" * 60)
    print("BLUE = You")
    print("RED  = Frozen Trial 1 Seed 2")
    print()
    print("Move BLUE using your mouse/touchpad.")
    print("Press ESC to quit.")
    print()

    blue_goals = 0
    red_goals = 0

    running = True

    while running:

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

        blue_action = get_blue_action(env)

        red_action = get_t1s2_action(env)

        combined_action = np.concatenate(
            [
                blue_action,
                red_action
            ]
        ).astype(np.float32)

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            combined_action
        )

        if terminated or truncated:

            goal = info.get(
                "goal",
                0
            )

            if goal == 1:

                blue_goals += 1

                print(
                    f"YOU scored!      "
                    f"Score: YOU {blue_goals} - "
                    f"{red_goals} T1S2"
                )

            elif goal == 2:

                red_goals += 1

                print(
                    f"T1S2 scored!     "
                    f"Score: YOU {blue_goals} - "
                    f"{red_goals} T1S2"
                )

            observation, info = env.reset()

    env.close()

    pygame.quit()

    print()
    print("=" * 60)
    print("FINAL SCORE")
    print("=" * 60)

    print(
        f"YOU {blue_goals} - "
        f"{red_goals} T1S2"
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()
