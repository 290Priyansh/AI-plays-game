import pygame
import random
import csv
import pickle
import pandas as pd
import os
import numpy as np
from collections import deque

# Game Constants
SPRING_CHANCE = 0.025
MAX_SPRING = 2
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 0.4
JUMP_STRENGTH = -11
PLATFORM_WIDTH = 60  # Match training model
PLATFORM_HEIGHT = 10  # Match training model
PLAYER_WIDTH = 50  # Match training model
PLAYER_HEIGHT = 50  # Match training model

# Paths to images
DOODLE_IMG = r'images/doodle.png'
SPRING_IMG = r'images/trampoline.jpg'
PLATFORM_IMG = r'images/platform.png'
SKY_IMG = r'images/sky.jpeg'

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Setup pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Doodle Jump")
clock = pygame.time.Clock()

# Load background image
BACKGROUND_IMG = pygame.image.load(SKY_IMG).convert()
BACKGROUND_IMG = pygame.transform.scale(BACKGROUND_IMG, (SCREEN_WIDTH, SCREEN_HEIGHT))


# Training Data Storage
training_data = []

# AI Model Loading
q_table = None
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print(" Q-table loaded successfully!")
    print(f"Q-table type: {type(q_table)}")
    print(f"Q-table size: {len(q_table) if isinstance(q_table, dict) else 'Unknown'}")
    if isinstance(q_table, dict):
        # Show sample states
        sample_states = list(q_table.keys())[:3]
        print(f"Sample states: {sample_states}")
except FileNotFoundError:
    print(" Q-table not found. AI mode will be disabled.")
except Exception as e:
    print(f"Error loading Q-table: {e}")

# Game state
AI_MODE = False
SHOW_AI_DECISION = True
ACTIONS = ["LEFT", "RIGHT", "NONE"]


# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Load doodle image properly
        self.image = None
        self.image = pygame.image.load(DOODLE_IMG).convert_alpha()
        self.image = pygame.transform.scale(self.image, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 250)
        self.vel_y = 0
        self.on_platform = False
        self.visited_platforms = deque(maxlen=5)  # Track last 5 visited platforms
        self.stuck_counter = 0
        self.last_y_position = self.rect.y
        self.no_progress_counter = 0
#updates the position of the player

    def update(self, action=None):
        self.vel_y += GRAVITY
        old_y = self.rect.y
        self.rect.y += self.vel_y

        if AI_MODE and action:
            # AI controls
            if action == "LEFT":
                self.rect.x -= 7
            elif action == "RIGHT":
                self.rect.x += 7
        else:
            # Manual controls
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.rect.x -= 7
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.rect.x += 7

        # Screen wrapping
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        if self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0

        # Check for vertical progress
        if abs(self.rect.y - self.last_y_position) < 5:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
            self.last_y_position = self.rect.y

        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

    def jump(self):
        self.vel_y = JUMP_STRENGTH
        self.on_platform = False

    def is_stuck_in_loop(self):
        """Check if player is stuck in a loop"""
        return (len(self.visited_platforms) >= 2 and
                len(set(self.visited_platforms)) <= 2) or self.no_progress_counter > 120

    def add_visited_platform(self, platform):
        """Add platform to visited history"""
        platform_id = (platform.rect.x, platform.rect.y)
        self.visited_platforms.append(platform_id)


# Platform class
class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, has_spring=False):
        super().__init__()
        try:
            self.image = pygame.image.load(PLATFORM_IMG).convert_alpha()
            self.image = pygame.transform.scale(self.image, (PLATFORM_WIDTH, PLATFORM_HEIGHT))
        except:
            # Create a simple platform if image not found
            self.image = pygame.Surface((PLATFORM_WIDTH, PLATFORM_HEIGHT))
            self.image.fill(GREEN)

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.has_spring = has_spring

    def update(self, dy):
        self.rect.y += dy
        if self.rect.top > SCREEN_HEIGHT:
            self.rect.x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)
            attempts = 0
            while attempts < 10:
                new_y = random.randint(-150, -50)
                if is_far_enough(new_y):
                    self.rect.y = new_y
                    break
                attempts += 1
            # Randomly add spring
            self.has_spring = random.random() < SPRING_CHANCE


# Spring class
class Spring(pygame.sprite.Sprite):
    def __init__(self, platform):
        super().__init__()

        self.image = pygame.image.load(SPRING_IMG).convert_alpha()
        self.image = pygame.transform.scale(self.image, (20, 10))

        self.rect = self.image.get_rect()
        self.platform = platform
        self.rect.centerx = platform.rect.centerx
        self.rect.bottom = platform.rect.top

    def update(self, dy):
        self.rect.y += dy
        self.rect.centerx = self.platform.rect.centerx
        self.rect.bottom = self.platform.rect.top


# gets the current platform location so that it can detect the next platform
def get_current_platform(player, platforms):
    """Find the platform the player is currently on or just landed on"""
    for plat in platforms:
        if plat.rect.colliderect(player.rect.move(0, 5)):
            return plat
    return None


def get_next_platform_only(player, platforms):
    """Get ONLY the next immediate platform above current position"""
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

    # Find the closest platform vertically
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
    """Get state with sticky platform targeting"""
    target_platform = None

    # Only assign a new target if conditions are met
    if (current_target_platform is None or
            player.on_platform or
            player.is_stuck_in_loop()):
        target_platform = get_next_platform_only(player, platforms)
    else:
        # Keep using the same target platform
        target_platform = current_target_platform

        # Check if target platform still exists
        if target_platform not in platforms:
            target_platform = get_next_platform_only(player, platforms)

    # Calculate state features - EXACTLY like training model
    if target_platform:
        dx = (target_platform.rect.centerx - player.rect.centerx) // 40
        dy = (target_platform.rect.y - player.rect.y) // 40
    else:
        dx, dy = 0, 0

    vel = int(player.vel_y)
    spring = detect_spring(player, platforms)
    next_dir = get_next_platform_direction(player, platforms)
    is_stuck = 1 if player.is_stuck_in_loop() else 0

    # Return 6-element tuple like training model
    return (dx, dy, vel, spring, next_dir, is_stuck), target_platform


def get_ai_action(state_tuple):
    """Get AI action using Q-table """
    if q_table is None:
        return "NONE"

    try:
        if isinstance(q_table, dict):
            if state_tuple in q_table:
                action_values = q_table[state_tuple]
                if isinstance(action_values, dict):
                    # Get the action with highest Q-value
                    best_action = max(action_values, key=action_values.get)
                    return best_action
                else:
                    return "NONE"
            else:
                # State not in Q-table, return random action
                return random.choice(ACTIONS)
        else:
            print(f"Unknown Q-table format: {type(q_table)}")
            return "NONE"
    except Exception as e:
        print(f"Error in AI decision: {e}")
        return "NONE"


def is_far_enough(new_y):
    for plat in platforms:
        if abs(plat.rect.y - new_y) < 100:
            return False
    return True


# Groups
all_sprites = pygame.sprite.Group()
platforms = pygame.sprite.Group()
springs = pygame.sprite.Group()


# UI Functions
def draw_ui():
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)

    # Score
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # Mode indicator
    mode_text = "AI MODE" if AI_MODE else "MANUAL MODE"
    mode_color = GREEN if AI_MODE else BLUE
    mode_surface = font.render(mode_text, True, mode_color)
    screen.blit(mode_surface, (10, 50))

    # Player status (only in AI mode)
    if AI_MODE:
        status_text = "STUCK" if player.is_stuck_in_loop() else "NORMAL"
        status_color = RED if player.is_stuck_in_loop() else GREEN
        status_surface = small_font.render(f"Status: {status_text}", True, status_color)
        screen.blit(status_surface, (10, 90))

    # Controls
    controls = [
        "SPACE: Toggle AI/Manual",
        "R: Restart Game",

    ]

    for i, control in enumerate(controls):
        control_surface = small_font.render(control, True, BLACK)
        screen.blit(control_surface, (10, SCREEN_HEIGHT - 80 + i * 20))


def draw_ai_decision(action, target_platform):
    """Draw AI decision info - only in AI mode"""
    if not SHOW_AI_DECISION or not AI_MODE:
        return
    # Draw line to target platform (only in AI mode)
    if target_platform and player in all_sprites:
        pygame.draw.line(screen, RED, player.rect.center, target_platform.rect.center, 2)
        pygame.draw.circle(screen, RED, target_platform.rect.center, 30, 2)


def reset_game():
    global score, all_sprites, platforms, springs, player, current_target_platform

    # Clear all sprites
    all_sprites.empty()
    platforms.empty()
    springs.empty()

    # Reset score
    score = 0
    current_target_platform = None

    # Create new player
    player = Player()
    all_sprites.add(player)

    # Create initial platforms - MATCH TRAINING MODEL
    initial_platform = Platform(SCREEN_WIDTH // 2 - PLATFORM_WIDTH // 2, SCREEN_HEIGHT - 60)
    platforms.add(initial_platform)
    all_sprites.add(initial_platform)

    # Generate initial platforms
    for i in range(1, 6):
        x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)
        y = SCREEN_HEIGHT - i * 120
        has_spring = random.random() < 0.2
        plat = Platform(x, y, has_spring)
        platforms.add(plat)
        all_sprites.add(plat)


# Initial setup
reset_game()
score = 0
font = pygame.font.SysFont(None, 36)
current_target_platform = None

# Game Loop
running = True
current_action = "NONE"

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if q_table is not None:
                    AI_MODE = not AI_MODE
                    print(f"Switched to {'AI' if AI_MODE else 'Manual'} mode")
                else:
                    print("Q-table not available!")
            elif event.key == pygame.K_r:
                reset_game()
                print("Game restarted!")

    # Get current state and action
    state_tuple, current_target_platform = get_state(player, platforms, current_target_platform)

    if AI_MODE:
        current_action = get_ai_action(state_tuple)
    else:
        # Capture manual key state for data collection
        keys = pygame.key.get_pressed()
        current_action = "NONE"
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            current_action = "LEFT"
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            current_action = "RIGHT"

    # Record training data (only in manual mode)
    if not AI_MODE:
        training_data.append((*state_tuple, current_action))

    # Update player
    player.update(current_action if AI_MODE else None)

    # Reset on_platform flag
    player.on_platform = False

    # Platform collision
    if player.vel_y > 0:
        for plat in platforms:
            if plat.rect.colliderect(player.rect) and player.rect.bottom <= plat.rect.bottom + 10:
                if plat.has_spring:
                    player.vel_y = JUMP_STRENGTH * 1.5
                else:
                    player.jump()

                player.on_platform = True
                player.add_visited_platform(plat)
                break

    # Scrolling
    scroll = 0
    if player.rect.top <= SCREEN_HEIGHT / 3:
        scroll = SCREEN_HEIGHT / 3 - player.rect.top
        player.rect.top = SCREEN_HEIGHT / 3
        score += int(scroll)

    # Update platforms
    for platform in platforms:
        platform.update(scroll)

    # Add new platforms
    highest_y = min([plat.rect.y for plat in platforms]) if platforms else 0
    if highest_y > 0:
        new_y = highest_y - random.randint(80, 120)
        new_x = random.randint(0, SCREEN_WIDTH - PLATFORM_WIDTH)
        has_spring = random.random() < 0.2
        new_platform = Platform(new_x, new_y, has_spring)
        platforms.add(new_platform)
        all_sprites.add(new_platform)

    # Drawing
    screen.blit(BACKGROUND_IMG, (0, 0))

    # Draw platforms
    for platform in platforms:
        screen.blit(platform.image, platform.rect)
        # Draw springs
        if platform.has_spring:
            spring_img = pygame.image.load(SPRING_IMG).convert_alpha()
            spring_img = pygame.transform.scale(spring_img, (40, 30))
            spring_rect = spring_img.get_rect()
            spring_rect.centerx = platform.rect.centerx
            spring_rect.bottom = platform.rect.top
            screen.blit(spring_img, spring_rect)

    # Draw player - ALWAYS use the image, no colored rectangles
    screen.blit(player.image, player.rect)

    # Draw UI
    draw_ui()

    # Draw AI decision info (only in AI mode)
    draw_ai_decision(current_action, current_target_platform)

    pygame.display.flip()

    # Game over
    if player not in all_sprites:
        if AI_MODE:
            reset_game()
        else:
            print("Game Over! Press R to restart or SPACE to switch to AI mode.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            reset_game()
                            waiting = False
                        elif event.key == pygame.K_SPACE and q_table is not None:
                            AI_MODE = True
                            reset_game()
                            waiting = False

pygame.quit()

#Save training data if collected
if training_data:
    with open("labeled_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dx", "dy", "vel", "spring", "next_dir", "is_stuck", "action"])
        writer.writerows(training_data)
    print(" Additional training data saved!")