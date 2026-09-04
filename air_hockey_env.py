# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:29:33 2026

@author: nibra
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class AirHockeyEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None):
    
        super().__init__()
    
        # ============================================================
        # ENVIRONMENT SETTINGS
        # ============================================================
    
        # Table dimensions
        self.width = 300
        self.height = 600
    
        # Border
        self.border = 30
    
        # Goal dimensions
        self.goal_width = 108
    
        # Object sizes
        self.puck_radius = 11
        self.paddle_radius = 21
    
        # Object masses
        self.puck_mass = 1.0
        self.paddle_mass = 1.5
    
        # ------------------------------------------------------------
        # ACTUAL GAME PHYSICS
        # ------------------------------------------------------------
    
        # These stay exactly as they were when T5S1 was trained.
        self.paddle_speed = 8.0
        self.max_puck_speed = 20.0
    
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
    
        # ============================================================
        # ACTION SPACE
        # ============================================================
    
        # Action:
        #
        # [
        #   paddle_1_x,
        #   paddle_1_y,
        #   paddle_2_x,
        #   paddle_2_y
        # ]
        #
        # Each number is between -1 and 1.
    
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
    
        # ============================================================
        # OBSERVATION SPACE
        # ============================================================
    
        # Observation:
        #
        # [
        #   puck_x,
        #   puck_y,
        #   puck_vx,
        #   puck_vy,
        #
        #   paddle_1_x,
        #   paddle_1_y,
        #   paddle_1_vx,
        #   paddle_1_vy,
        #
        #   paddle_2_x,
        #   paddle_2_y,
        #   paddle_2_vx,
        #   paddle_2_vy
        # ]
        #
        # IMPORTANT:
        # The actual physics still uses:
        #
        #     puck max speed   = 20
        #     paddle max speed = 8
        #
        # But the saved T5S1 PPO model expects the declared observation
        # space to use velocity bounds of:
        #
        #     puck   = +/-25
        #     paddle = +/-10
        #
        # These wider values are therefore only observation-space bounds.
        # They do NOT change the actual game speeds.
    
        low = np.array([
            0,
            0,
            -25,
            -25,
    
            0,
            0,
            -10,
            -10,
    
            0,
            0,
            -10,
            -10
        ], dtype=np.float32)
    
        high = np.array([
            self.width,
            self.height,
            25,
            25,
    
            self.width,
            self.height,
            10,
            10,
    
            self.width,
            self.height,
            10,
            10
        ], dtype=np.float32)
    
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
    
        # ============================================================
        # GAME OBJECTS
        # ============================================================
    
        self.puck_pos = None
        self.puck_vel = None
    
        self.paddle1_pos = None
        self.paddle2_pos = None
    
        self.paddle1_vel = None
        self.paddle2_vel = None
    
        # ============================================================
        # SLOW PUCK RESET
        # ============================================================
    
        self.slow_puck_steps = 0
    
        self.slow_puck_threshold = (
            0.10 * self.max_puck_speed
        )
    
        self.slow_puck_step_limit = int(
            1.5 * self.metadata["render_fps"]
        )
    
        self.skip_slow_timer = False
    
        # ============================================================
        # GOAL / RESTART TRACKING
        # ============================================================
    
        # Stores which player conceded the previous goal
        #
        # 1 = Player 1 / BLUE / bottom
        # 2 = Player 2 / RED / top
    
        self.last_conceded = None


    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
    
        # ------------------------------------------------------------
        # Position after a goal
        # ------------------------------------------------------------
    
        if self.last_conceded == 1:
    
            # BLUE / bottom player conceded.
            #
            # Halfway = 300
            # Puck    = 400  (first third)
            # BLUE    = 500  (second third)
    
            self.puck_pos = np.array([
                self.width / 2,
                self.height / 2 + self.height / 6
            ], dtype=np.float32)
    
            self.paddle1_pos = np.array([
                self.width / 2,
                self.height / 2 + self.height / 3
            ], dtype=np.float32)
    
            # RED returns to normal starting position
            self.paddle2_pos = np.array([
                self.width / 2,
                self.height * 0.25
            ], dtype=np.float32)
    
    
        elif self.last_conceded == 2:
    
            # RED / top player conceded.
            #
            # RED     = 100  (second third)
            # Puck    = 200  (first third)
            # Halfway = 300
    
            self.puck_pos = np.array([
                self.width / 2,
                self.height / 2 - self.height / 6
            ], dtype=np.float32)
    
            self.paddle2_pos = np.array([
                self.width / 2,
                self.height / 2 - self.height / 3
            ], dtype=np.float32)
    
            # BLUE returns to normal starting position
            self.paddle1_pos = np.array([
                self.width / 2,
                self.height * 0.75
            ], dtype=np.float32)
    
    
        else:
    
            # --------------------------------------------------------
            # Normal first reset
            # --------------------------------------------------------
    
            self.puck_pos = np.array([
                self.width / 2,
                self.height / 2
            ], dtype=np.float32)
    
            # Bottom paddle - Player 1
            self.paddle1_pos = np.array([
                self.width / 2,
                self.height * 0.75
            ], dtype=np.float32)
    
            # Top paddle - Player 2
            self.paddle2_pos = np.array([
                self.width / 2,
                self.height * 0.25
            ], dtype=np.float32)
    
    
        # ------------------------------------------------------------
        # Reset velocities
        # ------------------------------------------------------------
    
        self.puck_vel = np.array([
            0.0,
            0.0
        ], dtype=np.float32)
    
        self.paddle1_vel = np.array([
            0.0,
            0.0
        ], dtype=np.float32)
    
        self.paddle2_vel = np.array([
            0.0,
            0.0
        ], dtype=np.float32)
    
    
        # Reset slow-puck counter
        self.slow_puck_steps = 0
    
        # We've now used the previous goal information.
        self.last_conceded = None
    
    
        observation = self._get_observation()
    
        info = {}
    
        if self.render_mode == "human":
            self.render()
    
        return observation, info
        
    #How the paddles and puck looks
    def _draw_air_holes(self):

        hole_colour = (190, 200, 205)
    
        spacing = 24
        hole_radius = 2
        margin = 20
    
        row = 0
        y = margin
    
        while y < self.height - margin:
    
            if row % 2 == 0:
                x_start = margin
            else:
                x_start = margin + spacing // 2
    
            x = x_start
    
            while x < self.width - margin:
    
                pygame.draw.circle(
                    self.screen,
                    hole_colour,
                    (
                        int(x + self.border),
                        int(y + self.border)
                    ),
                    hole_radius
                )
    
                x += spacing
    
            y += spacing * 0.866
            row += 1
        
    def _draw_checker_piece(self, centre, radius, colour):

        x = int(centre[0] + self.border)
        y = int(centre[1] + self.border)
    
        # Dark outer edge
        pygame.draw.circle(
            self.screen,
            (30, 30, 30),
            (x, y),
            radius + 3
        )
    
        # Main colour
        pygame.draw.circle(
            self.screen,
            colour,
            (x, y),
            radius
        )
    
        # Darker inner ring
        darker_colour = tuple(
            max(0, c - 50) for c in colour
        )
    
        pygame.draw.circle(
            self.screen,
            darker_colour,
            (x, y),
            int(radius * 0.72),
            3
        )
    
        # Highlight
        highlight_colour = tuple(
            min(255, c + 80) for c in colour
        )
    
        pygame.draw.arc(
            self.screen,
            highlight_colour,
            (
                x - radius + 6,
                y - radius + 6,
                2 * radius - 12,
                2 * radius - 12
            ),
            3.5,
            5.4,
            3
        )

    


    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        action = np.clip(action, -1, 1)
    
        # ------------------------------------------------------------
        # Move paddles
        # ------------------------------------------------------------
    
        requested_paddle1_vel = (
            action[0:2] * self.paddle_speed
        )
    
        requested_paddle2_vel = (
            action[2:4] * self.paddle_speed
        )
    
        # Save old positions so actual movement can be calculated.
        old_paddle1_pos = self.paddle1_pos.copy()
        old_paddle2_pos = self.paddle2_pos.copy()
    
        # Move each paddle, but prevent it from squeezing
        # the puck into a wall.
        self.paddle1_pos = self._move_paddle_safely(
            self.paddle1_pos,
            requested_paddle1_vel,
            player=1
        )
    
        self.paddle2_pos = self._move_paddle_safely(
            self.paddle2_pos,
            requested_paddle2_vel,
            player=2
        )
    
        # Velocity should represent how far the paddle ACTUALLY moved,
        # rather than how far it tried to move.
        self.paddle1_vel = (
            self.paddle1_pos
            - old_paddle1_pos
        )
    
        self.paddle2_vel = (
            self.paddle2_pos
            - old_paddle2_pos
        )
    
        # Final safety check for normal paddle boundaries.
        self._limit_paddles()
    
    
        # ------------------------------------------------------------
        # Move puck
        # ------------------------------------------------------------
    
        self.puck_pos += self.puck_vel
    
    
        # ------------------------------------------------------------
        # Wall collisions
        # ------------------------------------------------------------
    
        self._handle_wall_collisions()
    
    
        # ------------------------------------------------------------
        # Paddle collisions
        # ------------------------------------------------------------
    
        self._handle_paddle_collision(
            self.paddle1_pos,
            self.paddle1_vel
        )
    
        self._handle_paddle_collision(
            self.paddle2_pos,
            self.paddle2_vel
        )
    
    
        # ------------------------------------------------------------
        # Re-check walls after paddle collisions
        # ------------------------------------------------------------
        # A paddle collision can push the puck partly into a wall.
        # This immediately forces it back inside the table.
    
        self._handle_wall_collisions()
    
    
        # ------------------------------------------------------------
        # Check goals
        # ------------------------------------------------------------
    
        goal = self._check_goal()
    
        terminated = False
        reward = 0.0
    
    
        # ------------------------------------------------------------
        # BLUE / Player 1 scores
        # ------------------------------------------------------------
    
        if goal == 1:
    
            reward = 1.0
            terminated = True
    
            # RED / Player 2 conceded.
            self.last_conceded = 2
    
            # The next puck spawn should NOT have a slow-puck timer
            # until someone actually gets the puck moving.
            self.slow_puck_steps = 0
            self.skip_slow_timer = True
    
    
        # ------------------------------------------------------------
        # RED / Player 2 scores
        # ------------------------------------------------------------
    
        elif goal == 2:
    
            reward = -1.0
            terminated = True
    
            # BLUE / Player 1 conceded.
            self.last_conceded = 1
    
            # The next puck spawn should NOT have a slow-puck timer
            # until someone actually gets the puck moving.
            self.slow_puck_steps = 0
            self.skip_slow_timer = True
    
    
        # ------------------------------------------------------------
        # No goal - check whether puck is moving too slowly
        # ------------------------------------------------------------
    
        else:
    
            current_puck_speed = np.linalg.norm(
                self.puck_vel
            )
    
            # --------------------------------------------------------
            # Immediately after a goal respawn:
            #
            # Ignore the slow-puck timer until the puck has first
            # reached at least 10% of max speed.
            # --------------------------------------------------------
    
            if self.skip_slow_timer:
    
                if (
                    current_puck_speed
                    >=
                    self.slow_puck_threshold
                ):
    
                    self.skip_slow_timer = False
                    self.slow_puck_steps = 0
    
    
            # --------------------------------------------------------
            # Normal slow-puck detection
            # --------------------------------------------------------
    
            else:
    
                if (
                    current_puck_speed
                    <
                    self.slow_puck_threshold
                ):
    
                    self.slow_puck_steps += 1
    
                else:
    
                    # Puck sped back up, so the 1.5-second count
                    # starts again from zero next time it slows down.
                    self.slow_puck_steps = 0
    
    
            # --------------------------------------------------------
            # Puck has stayed below 10% speed for 1.5 seconds
            # --------------------------------------------------------
    
            if (
                self.slow_puck_steps
                >=
                self.slow_puck_step_limit
            ):
    
                # Reset puck to exact centre of table.
                self.puck_pos = np.array([
                    self.width / 2,
                    self.height / 2
                ], dtype=np.float32)
    
                self.puck_vel = np.array([
                    0.0,
                    0.0
                ], dtype=np.float32)
    
                self.slow_puck_steps = 0
    
    
        # ------------------------------------------------------------
        # Finish step
        # ------------------------------------------------------------
    
        truncated = False
    
        observation = self._get_observation()
    
        info = {
            "goal": goal
        }
    
        if self.render_mode == "human":
            self.render()
    
        return observation, reward, terminated, truncated, info


    # ================================================================
    # OBSERVATION
    # ================================================================

    def _get_observation(self):

        return np.array([
            self.puck_pos[0],
            self.puck_pos[1],
            self.puck_vel[0],
            self.puck_vel[1],
    
            self.paddle1_pos[0],
            self.paddle1_pos[1],
            self.paddle1_vel[0],
            self.paddle1_vel[1],
    
            self.paddle2_pos[0],
            self.paddle2_pos[1],
            self.paddle2_vel[0],
            self.paddle2_vel[1]
    
        ], dtype=np.float32)


    # ================================================================
    # PADDLE BOUNDARIES
    # ================================================================
    
    def _move_paddle_safely(
        self,
        paddle_pos,
        paddle_vel,
        player
    ):
    
        old_pos = paddle_pos.copy()
    
        desired_pos = (
            old_pos
            + paddle_vel
        )
    
        paddle_r = self.paddle_radius
        puck_r = self.puck_radius
    
        # ------------------------------------------------------------
        # First apply the normal paddle boundaries
        # ------------------------------------------------------------
    
        desired_pos[0] = np.clip(
            desired_pos[0],
            paddle_r,
            self.width - paddle_r
        )
    
        if player == 1:
    
            desired_pos[1] = np.clip(
                desired_pos[1],
                self.height / 2 + paddle_r,
                self.height - paddle_r
            )
    
        else:
    
            desired_pos[1] = np.clip(
                desired_pos[1],
                paddle_r,
                self.height / 2 - paddle_r
            )
    
        minimum_distance = (
            paddle_r
            + puck_r
        )
    
        # ------------------------------------------------------------
        # Check whether a paddle position would trap the puck
        # ------------------------------------------------------------
    
        def position_is_safe(test_pos):
    
            difference = (
                self.puck_pos
                - test_pos
            )
    
            distance = np.linalg.norm(
                difference
            )
    
            # Paddle does not touch puck at all.
            if distance >= minimum_distance:
                return True
    
            # Completely overlapping centres is never safe.
            if distance <= 1e-8:
                return False
    
            normal = (
                difference
                / distance
            )
    
            # This is where the existing collision code would
            # push the puck to remove the overlap.
            corrected_puck_pos = (
                test_pos
                + normal * minimum_distance
            )
    
            x = corrected_puck_pos[0]
            y = corrected_puck_pos[1]
    
            # --------------------------------------------------------
            # Left / right walls
            # --------------------------------------------------------
    
            if x - puck_r < 0:
                return False
    
            if x + puck_r > self.width:
                return False
    
            # --------------------------------------------------------
            # Goal opening
            # --------------------------------------------------------
    
            goal_left = (
                self.width
                - self.goal_width
            ) / 2
    
            goal_right = (
                self.width
                + self.goal_width
            ) / 2
    
            puck_fits_goal = (
                x - puck_r >= goal_left
                and
                x + puck_r <= goal_right
            )
    
            # --------------------------------------------------------
            # Top wall
            # --------------------------------------------------------
    
            if y - puck_r < 0:
    
                if not puck_fits_goal:
                    return False
    
            # --------------------------------------------------------
            # Bottom wall
            # --------------------------------------------------------
    
            if y + puck_r > self.height:
    
                if not puck_fits_goal:
                    return False
    
            return True
    
        # ------------------------------------------------------------
        # Normal movement is fine
        # ------------------------------------------------------------
    
        if position_is_safe(
            desired_pos
        ):
            return desired_pos
    
        # ------------------------------------------------------------
        # Requested movement would squash puck into a wall.
        #
        # Find the furthest safe point along that movement instead.
        # ------------------------------------------------------------
    
        low = 0.0
        high = 1.0
    
        safe_pos = old_pos.copy()
    
        for _ in range(12):
    
            middle = (
                low + high
            ) / 2
    
            test_pos = (
                old_pos
                + middle
                * (
                    desired_pos
                    - old_pos
                )
            )
    
            if position_is_safe(
                test_pos
            ):
    
                safe_pos = test_pos
                low = middle
    
            else:
    
                high = middle
    
        return safe_pos.astype(
            np.float32
        )

    
    def _limit_paddles(self):

        r = self.paddle_radius
    
        # Player 1 stays in BOTTOM half
        self.paddle1_pos[0] = np.clip(
            self.paddle1_pos[0],
            r,
            self.width - r
        )
    
        self.paddle1_pos[1] = np.clip(
            self.paddle1_pos[1],
            self.height / 2 + r,
            self.height - r
        )
    
        # Player 2 stays in TOP half
        self.paddle2_pos[0] = np.clip(
            self.paddle2_pos[0],
            r,
            self.width - r
        )
    
        self.paddle2_pos[1] = np.clip(
            self.paddle2_pos[1],
            r,
            self.height / 2 - r
        )


    # ================================================================
    # WALL COLLISIONS
    # ================================================================

    def _handle_wall_collisions(self):
    
        r = self.puck_radius
    
        # ------------------------------------------------------------
        # LEFT WALL
        # ------------------------------------------------------------
    
        if self.puck_pos[0] - r <= 0:
    
            # Keep puck physically inside the table.
            self.puck_pos[0] = r
    
            # Only bounce if puck is actually travelling INTO the wall.
            if self.puck_vel[0] < 0:
                self.puck_vel[0] *= -1
    
        # ------------------------------------------------------------
        # RIGHT WALL
        # ------------------------------------------------------------
    
        elif self.puck_pos[0] + r >= self.width:
    
            self.puck_pos[0] = (
                self.width - r
            )
    
            # Only bounce if travelling INTO the wall.
            if self.puck_vel[0] > 0:
                self.puck_vel[0] *= -1
    
        # ------------------------------------------------------------
        # Goal opening
        # ------------------------------------------------------------
    
        goal_left = (
            self.width - self.goal_width
        ) / 2
    
        goal_right = (
            self.width + self.goal_width
        ) / 2
    
        x = self.puck_pos[0]
    
        # Whole puck must fit through the goal.
        puck_inside_goal_width = (
            x - r >= goal_left
            and
            x + r <= goal_right
        )
    
        # ------------------------------------------------------------
        # TOP WALL
        # ------------------------------------------------------------
    
        if (
            self.puck_pos[1] - r <= 0
            and not puck_inside_goal_width
        ):
    
            self.puck_pos[1] = r
    
            # Only bounce if travelling upwards INTO the wall.
            if self.puck_vel[1] < 0:
                self.puck_vel[1] *= -1
    
        # ------------------------------------------------------------
        # BOTTOM WALL
        # ------------------------------------------------------------
    
        elif (
            self.puck_pos[1] + r >= self.height
            and not puck_inside_goal_width
        ):
    
            self.puck_pos[1] = (
                self.height - r
            )
    
            # Only bounce if travelling downwards INTO the wall.
            if self.puck_vel[1] > 0:
                self.puck_vel[1] *= -1


    # ================================================================
    # PADDLE COLLISIONS
    # ================================================================

    def _handle_paddle_collision(self, paddle_pos, paddle_vel):

        # Vector pointing from paddle centre to puck centre
        difference = self.puck_pos - paddle_pos
    
        distance = np.linalg.norm(difference)
    
        minimum_distance = (
            self.puck_radius +
            self.paddle_radius
        )
    
        # Check whether paddle and puck overlap
        if distance < minimum_distance:
    
            # Avoid division by zero
            if distance == 0:
                normal = np.array(
                    [1.0, 0.0],
                    dtype=np.float32
                )
            else:
                normal = difference / distance
    
            # Move puck outside the paddle
            self.puck_pos = (
                paddle_pos +
                normal * minimum_distance
            )
    
            # ------------------------------------------------------
            # Find velocity along collision direction
            # ------------------------------------------------------
    
            puck_normal_speed = np.dot(
                self.puck_vel,
                normal
            )
    
            paddle_normal_speed = np.dot(
                paddle_vel,
                normal
            )
    
            # Relative speed between puck and paddle
            relative_speed = (
                puck_normal_speed -
                paddle_normal_speed
            )
    
            # Only collide if they're moving towards each other
            if relative_speed < 0:
    
                puck_mass = self.puck_mass
                paddle_mass = self.paddle_mass
    
                # Elastic collision:
                # calculate new puck speed along the normal
                new_puck_normal_speed = (
                    (
                        (puck_mass - paddle_mass)
                        /
                        (puck_mass + paddle_mass)
                    )
                    * puck_normal_speed
                    +
                    (
                        (2 * paddle_mass)
                        /
                        (puck_mass + paddle_mass)
                    )
                    * paddle_normal_speed
                )
    
                # --------------------------------------------------
                # Replace only puck's normal velocity
                # --------------------------------------------------
    
                self.puck_vel += (
                    new_puck_normal_speed
                    - puck_normal_speed
                ) * normal
    
                # --------------------------------------------------
                # Maximum puck speed
                # --------------------------------------------------
    
                speed = np.linalg.norm(
                    self.puck_vel
                )
    
                if speed > self.max_puck_speed:
    
                    self.puck_vel = (
                        self.puck_vel
                        / speed
                        * self.max_puck_speed
                    )

    # ================================================================
    # GOALS
    # ================================================================

    def _check_goal(self):

        goal_left = (
            self.width - self.goal_width
        ) / 2
    
        goal_right = (
            self.width + self.goal_width
        ) / 2
    
        puck_in_goal = (
            goal_left
            < self.puck_pos[0]
            < goal_right
        )
    
        # Player 1 (bottom) scores in TOP goal
        if self.puck_pos[1] < 0 and puck_in_goal:
            return 1
    
        # Player 2 (top) scores in BOTTOM goal
        if self.puck_pos[1] > self.height and puck_in_goal:
            return 2
    
        return 0


    # ================================================================
    # RENDERING
    # ================================================================

    def render(self):

        if self.screen is None:

            pygame.init()

            self.screen = pygame.display.set_mode(
                (
                    self.width + 2 * self.border,
                    self.height + 2 * self.border
                )
            )

            pygame.display.set_caption(
                "Air Hockey Environment"
            )

            self.clock = pygame.time.Clock()

        # Colours
        WHITE = (240, 240, 240)
        BLACK = (20, 20, 20)
        BLUE = (50, 100, 255)
        RED = (255, 70, 70)
        GREY = (100, 100, 100)
        PUCK_COLOUR = (55, 55, 55)

        # Outer border/background
        self.screen.fill((40, 40, 45))
        
        # Actual air hockey surface
        pygame.draw.rect(
            self.screen,
            WHITE,
            (
                self.border,
                self.border,
                self.width,
                self.height
            )
        )
        
        self._draw_air_holes()
        
        

        # ------------------------------------------------------------
        # Centre line
        # ------------------------------------------------------------

        pygame.draw.line(
            self.screen,
            GREY,
            (
                self.border,
                self.height // 2 + self.border
            ),
            (
                self.width + self.border,
                self.height // 2 + self.border
            ),
            3
        )

        # Centre circle
        pygame.draw.circle(
            self.screen,
            GREY,
            (
                self.width // 2 + self.border,
                self.height // 2 + self.border
            ),
            42,
            3
        )

        # ------------------------------------------------------------
        # Goals
        # ------------------------------------------------------------

        goal_left = int(
            (self.width - self.goal_width) / 2
        )
        
        goal_right = int(
            (self.width + self.goal_width) / 2
        )
        
        # Top goal recess
        pygame.draw.rect(
            self.screen,
            BLACK,
            (
                goal_left + self.border,
                0,
                self.goal_width,
                self.border
            )
        )
        
        # Bottom goal recess
        pygame.draw.rect(
            self.screen,
            BLACK,
            (
                goal_left + self.border,
                self.height + self.border,
                self.goal_width,
                self.border
            )
        )
                

        # ------------------------------------------------------------
        # Paddles
        # ------------------------------------------------------------

        self._draw_checker_piece(
        self.paddle1_pos,
        self.paddle_radius,
        BLUE
    )
    
        self._draw_checker_piece(
        self.paddle2_pos,
        self.paddle_radius,
        RED
    )
            

        # ------------------------------------------------------------
        # Puck
        # ------------------------------------------------------------

        self._draw_checker_piece(
        self.puck_pos,
        self.puck_radius,
        PUCK_COLOUR
    )

        pygame.display.flip()

        self.clock.tick(
            self.metadata["render_fps"]
        )


    # ================================================================
    # CLOSE
    # ================================================================

    def close(self):

        if self.screen is not None:
            pygame.quit()

            self.screen = None
            self.clock = None