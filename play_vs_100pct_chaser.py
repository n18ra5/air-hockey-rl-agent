# -*- coding: utf-8 -*-
"""
PLAY AGAINST 100% CHASER
========================

BLUE:
    You control the paddle with your mouse / touchpad.

RED:
    100%-speed chaser.

Chaser behaviour:
    - If any part of the puck is on RED's half, RED chases the puck.
    - Otherwise RED returns to the middle of its own half.
    - RED moves at 100% of normal paddle speed.

Controls:
    Mouse / touchpad = move BLUE paddle
    ESC              = quit
"""

import numpy as np
import pygame

from air_hockey_env import AirHockeyEnv


# ================================================================
# BLUE HUMAN CONTROL
# ================================================================

def get_blue_action(env):
    """
    Convert the mouse/touchpad position into BLUE's movement action.
    """

    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Convert window coordinates into table coordinates.
    target = np.array(
        [
            mouse_x - env.border,
            mouse_y - env.border
        ],
        dtype=np.float32
    )

    # BLUE must remain inside its own bottom half.
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
        -
        env.paddle1_pos
    )

    # Convert desired movement into Gym action values from -1 to +1.
    blue_action = np.clip(
        displacement / env.paddle_speed,
        -1.0,
        1.0
    )

    return blue_action.astype(np.float32)


# ================================================================
# RED 100% CHASER
# ================================================================

def get_red_chaser_action(env):
    """
    RED uses the same 100%-speed chaser logic used in training.

    If any part of the puck reaches RED's half, RED chases it.
    Otherwise RED returns to the centre of its own half.
    """

    puck_on_red_side = (
        env.puck_pos[1] - env.puck_radius
        <=
        env.height / 2
    )

    if puck_on_red_side:

        target = env.puck_pos.copy()

    else:

        target = np.array(
            [
                env.width / 2,
                env.height * 0.25
            ],
            dtype=np.float32
        )

    direction = (
        target
        -
        env.paddle2_pos
    )

    distance = float(
        np.linalg.norm(direction)
    )

    if distance <= 1e-8:

        return np.zeros(
            2,
            dtype=np.float32
        )

    # Unit vector = full action magnitude.
    # Therefore RED moves at 100% paddle speed.
    red_action = (
        direction
        /
        distance
    )

    return red_action.astype(np.float32)


# ================================================================
# MAIN GAME
# ================================================================

def main():

    env = AirHockeyEnv(
        render_mode="human"
    )

    observation, info = env.reset()

    pygame.display.set_caption(
        "Air Hockey - YOU vs 100% Chaser"
    )

    blue_goals = 0
    red_goals = 0

    print()
    print("=" * 60)
    print("AIR HOCKEY - YOU vs 100% CHASER")
    print("=" * 60)
    print("BLUE = You")
    print("RED  = 100% chaser")
    print()
    print("Move BLUE using your mouse/touchpad.")
    print("Press ESC to quit.")
    print()

    running = True

    while running:

        # ------------------------------------------------------------
        # PYGAME EVENTS
        # ------------------------------------------------------------

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif (
                event.type == pygame.KEYDOWN
                and
                event.key == pygame.K_ESCAPE
            ):
                running = False

        if not running:
            break

        # ------------------------------------------------------------
        # GET BOTH ACTIONS
        # ------------------------------------------------------------

        blue_action = get_blue_action(env)

        red_action = get_red_chaser_action(env)

        # Original environment expects:
        #
        # [
        #   blue_x,
        #   blue_y,
        #   red_x,
        #   red_y
        # ]

        combined_action = np.concatenate(
            [
                blue_action,
                red_action
            ]
        ).astype(np.float32)

        # ------------------------------------------------------------
        # STEP GAME
        # ------------------------------------------------------------

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            combined_action
        )

        # ------------------------------------------------------------
        # GOAL
        # ------------------------------------------------------------

        if terminated or truncated:

            goal = info.get(
                "goal",
                0
            )

            if goal == 1:

                blue_goals += 1

                print(
                    f"YOU scored!     "
                    f"Score: YOU {blue_goals} - {red_goals} CHASER"
                )

            elif goal == 2:

                red_goals += 1

                print(
                    f"CHASER scored!  "
                    f"Score: YOU {blue_goals} - {red_goals} CHASER"
                )

            # Start the next point.
            observation, info = env.reset()

    # ================================================================
    # CLOSE
    # ================================================================

    env.close()
    pygame.quit()

    print()
    print("=" * 60)
    print(
        f"FINAL SCORE: YOU {blue_goals} - {red_goals} CHASER"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
