import pygame
import random
import numpy as np
import sys
import pickle
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load existing Q-table if it exists
q_table = {}
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("Loaded existing Q-table.")
else:
    print("Starting with empty Q-table.")

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PLAYER_WIDTH = 50
PLAYER_HEIGHT = 50
PLATFORM_WIDTH = 60
PLATFORM_HEIGHT = 10
GRAVITY = 0.4
JUMP_STRENGTH = -11
FPS = 60

# Load Supervised Model
with open("supervised_model.pkl", "rb") as f:
    clf, *_ = pickle.load(f)

reverse_map = {0: "LEFT", 1: "RIGHT", 2: "NONE"}
ACTIONS = ["LEFT", "RIGHT", "NONE"]
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 0.1


class Player:
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.vel_y = 0
        self.on_platform = False
        self.visited_platforms = deque(maxlen=5)  # Track last 5 visited platforms
        self.stuck_counter = 0
        self.last_y_position = self.rect.y
        self.no_progress_counter = 0

    def update(self, action):
        self.vel_y += GRAVITY
        old_y = self.rect.y
        self.rect.y += self.vel_y

        if action == "LEFT":
            self.rect.x -= 7
        elif action == "RIGHT":
            self.rect.x += 7

        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        elif self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0

        # Check for vertical progress
        if abs(self.rect.y - self.last_y_position) < 5:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
            self.last_y_position = self.rect.y

    def jump(self):
        self.vel_y = JUMP_STRENGTH
        self.on_platform = False

    def is_stuck_in_loop(self):
        """Check if player is stuck in a loop - Modified to check for 2 or 3 platforms"""
        return (len(self.visited_platforms) >= 2 and
                len(set(self.visited_platforms)) <= 2) or self.no_progress_counter > 120

    def add_visited_platform(self, platform):
        """Add platform to visited history"""
        platform_id = (platform.rect.x, platform.rect.y)
        self.visited_platforms.append(platform_id)


class Platform:
    def __init__(self, x, y, has_spring=False):
        self.rect = pygame.Rect(x, y, PLATFORM_WIDTH, PLATFORM_HEIGHT)
        self.has_spring = has_spring


def get_current_platform(player, platforms):
    """Find the platform the player is currently on or just landed on"""
    for plat in platforms:
        if plat.rect.colliderect(player.rect.move(0, 5)):
            return plat
    return None


def get_next_platform_only(player, platforms):
    """Get ONLY the next immediate platform above current position, avoiding recently visited ones"""
    current_platform = get_current_platform(player, platforms)

    # Get all platforms above the player within jump range
    reachable_platforms = [
        p for p in platforms
        if 0 < (player.rect.y - p.rect.y) <= 150  # Above player and within jump range
    ]

    if not reachable_platforms:
        return None

    # Filter out recently visited platforms if player is stuck
    if player.is_stuck_in_loop() and len(reachable_platforms) > 1:
        visited_ids = set(player.visited_platforms)
        unvisited_platforms = [
            p for p in reachable_platforms
            if (p.rect.x, p.rect.y) not in visited_ids
        ]
        if unvisited_platforms:
            reachable_platforms = unvisited_platforms

    # Find the closest platform vertically (the immediate next one)
    next_platform = min(reachable_platforms, key=lambda p: abs(player.rect.y - p.rect.y))

    return next_platform


def detect_spring(player, platforms):
    for plat in platforms:
        if hasattr(plat, "has_spring") and plat.has_spring:
            spring_rect = pygame.Rect(plat.rect.centerx - 10, plat.rect.y - 10, 20, 10)
            if spring_rect.colliderect(player.rect):
                return 1
    return 0


def get_next_platform_direction(player, platforms, max_jump_height=150):
    next_platforms = [
        p for p in platforms if p.rect.y < player.rect.y and player.rect.y - p.rect.y <= max_jump_height
    ]
    if not next_platforms:
        return "NONE"
    next_plat = min(next_platforms, key=lambda p: abs(player.rect.y - p.rect.y))
    if next_plat.rect.centerx < player.rect.centerx - 10:
        return "LEFT"
    elif next_plat.rect.centerx > player.rect.centerx + 10:
        return "RIGHT"
    else:
        return "CENTER"


def get_state(player, platforms, current_target_platform):
    """
    Get state with sticky platform targeting and loop detection
    """
    target_platform = None

    # Only assign a new target if:
    # 1. No current target, OR
    # 2. Player is on a platform (just landed), OR
    # 3. Player is stuck in a loop
    if (current_target_platform is None or
            player.on_platform or
            player.is_stuck_in_loop()):
        target_platform = get_next_platform_only(player, platforms)
    else:
        # Keep using the same target platform
        target_platform = current_target_platform

        # Check if target platform still exists (might have scrolled off screen)
        if target_platform not in platforms:
            print("Target platform no longer exists, finding new target")
            target_platform = get_next_platform_only(player, platforms)

    # Calculate state features
    if target_platform:
        dx = (target_platform.rect.centerx - player.rect.centerx) // 40
        dy = (target_platform.rect.y - player.rect.y) // 40
    else:
        dx, dy = 0, 0
        print(" No target platform available")

    vel = int(player.vel_y)
    spring = detect_spring(player, platforms)
    next_dir = get_next_platform_direction(player, platforms)

    # Add loop detection to state
    is_stuck = 1 if player.is_stuck_in_loop() else 0

    return (dx, dy, vel, spring, next_dir, is_stuck), target_platform


def choose_action(state, player):
    if state not in q_table:
        q_table[state] = {a: 0 for a in ACTIONS}

    # If stuck in loop, force exploration (random action)
    if player.is_stuck_in_loop():
        return random.choice(ACTIONS)

    use_supervised = random.random() < 0.5
    if use_supervised:
        dx, dy, vel, spring, _, _ = state
        state_df = pd.DataFrame([[dx, dy, vel, spring]], columns=["dx", "dy", "vel_y", "spring"])
        action_num = clf.predict(state_df)[0]
        return reverse_map[action_num]
    else:
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        return max(q_table[state], key=q_table[state].get)


def update_q_table(prev_state, action, reward, next_state):
    if prev_state not in q_table:
        q_table[prev_state] = {a: 0 for a in ACTIONS}
    if next_state not in q_table:
        q_table[next_state] = {a: 0 for a in ACTIONS}
    max_future = max(q_table[next_state].values())
    old_value = q_table[prev_state][action]
    new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT * max_future)
    q_table[prev_state][action] = new_value


def save_q_table():
    """Save Q-table to file"""
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved!")


def train(episodes=500):  # Increased episodes to 500
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    action_counts = defaultdict(int)

    for ep in range(episodes):
        render = ep % 1 == 0
        player = Player()
        platforms = [Platform(SCREEN_WIDTH // 2 - PLATFORM_WIDTH // 2, SCREEN_HEIGHT - 60)]

        # Generate initial platforms
        for i in range(1, 6):
            x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)
            y = SCREEN_HEIGHT - i * 120
            has_spring = random.random() < 0.2
            platforms.append(Platform(x, y, has_spring))

        score = 0
        run = True
        current_target_platform = None
        prev_state, current_target_platform = get_state(player, platforms, current_target_platform)

        last_platform_y = None
        consecutive_same_platform = 0
        highest_reached = player.rect.y

        while run:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if render:
                screen.fill((255, 255, 255))

            action = choose_action(prev_state, player)
            action_counts[action] += 1
            player.update(action)

            reward = 1
            player.on_platform = False

            # Platform collision detection
            if player.vel_y > 0:
                for plat in platforms:
                    if plat.rect.top > SCREEN_HEIGHT:
                        continue
                    if player.rect.colliderect(plat.rect) and player.rect.bottom <= plat.rect.bottom + 10:
                        if plat.has_spring:
                            player.vel_y = JUMP_STRENGTH * 1.5
                        else:
                            player.jump()

                        player.on_platform = True
                        player.add_visited_platform(plat)  # Track visited platform

                        # Give bonus reward for reaching the targeted platform
                        if current_target_platform == plat:
                            reward += 15

                        # Check for consecutive same platform visits
                        if last_platform_y == plat.rect.y:
                            consecutive_same_platform += 1
                        else:
                            consecutive_same_platform = 0
                        last_platform_y = plat.rect.y

                        # Reward for reaching new heights
                        if player.rect.y < highest_reached:
                            reward += 5
                            highest_reached = player.rect.y

                        break

            # Enhanced penalties for looping behavior
            if consecutive_same_platform >= 3:
                reward -= 20

            if player.is_stuck_in_loop():
                reward -= 15

            # Reward movement actions slightly
            if action in ["LEFT", "RIGHT"]:
                reward += 0.5

            # Scrolling logic
            if player.rect.top < SCREEN_HEIGHT / 3:
                scroll = SCREEN_HEIGHT / 3 - player.rect.top
                reward += scroll / 5
                player.rect.top = SCREEN_HEIGHT / 3
                for plat in platforms:
                    plat.rect.y += scroll

                # Replace off-screen platforms
                for i in range(len(platforms)):
                    if platforms[i].rect.top > SCREEN_HEIGHT:
                        highest_y = min(p.rect.y for p in platforms)
                        new_y = highest_y - random.randint(80, 120)
                        platforms[i] = Platform(
                            random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH),
                            new_y,
                            random.random() < 0.2
                        )
                        score += 5

            # Rendering
            if render:
                # Draw player (red if stuck, blue otherwise)
                color = (255, 0, 0) if player.is_stuck_in_loop() else (0, 0, 255)
                pygame.draw.rect(screen, color, player.rect)

                for plat in platforms:
                    pygame.draw.rect(screen, (0, 255, 0), plat.rect)
                    if plat.has_spring:
                        spring_rect = pygame.Rect(plat.rect.centerx - 10, plat.rect.y - 10, 20, 10)
                        pygame.draw.rect(screen, (255, 0, 255), spring_rect)

                # Draw target platform
                if current_target_platform:
                    pygame.draw.line(screen, (255, 0, 0), player.rect.center,
                                     current_target_platform.rect.center, 3)
                    pygame.draw.circle(screen, (255, 0, 0), current_target_platform.rect.center, 35, 2)

                # Draw visited platforms in different color
                for visited_id in player.visited_platforms:
                    for plat in platforms:
                        if (plat.rect.x, plat.rect.y) == visited_id:
                            pygame.draw.rect(screen, (150, 150, 150), plat.rect, 3)

                pygame.display.flip()

            # Check for game over
            if player.rect.top > SCREEN_HEIGHT:
                reward = -50
                run = False
                continue

            # Get next state
            next_state, current_target_platform = get_state(player, platforms, current_target_platform)
            update_q_table(prev_state, action, reward, next_state)
            prev_state = next_state

        print(f"Episode {ep + 1} - Score: {score}")

        # Save Q-table every 10 episodes
        if (ep + 1) % 10 == 0:
            save_q_table()
            print(f"Progress saved at episode {ep + 1}")

    # Final save
    save_q_table()

    # Display action frequencies
    print("\nAction Frequencies:")
    for action in ACTIONS:
        print(f"{action}: {action_counts[action]}")
    plt.bar(action_counts.keys(), action_counts.values(), color=["red", "green", "blue"])
    plt.title("Action Frequency After Training")
    plt.xlabel("Actions")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
    pygame.quit()


if __name__ == "__main__":
    train(episodes=500)  # Increased episodes to 500

