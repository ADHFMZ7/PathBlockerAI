import pygame

class Renderer:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        self.clock = pygame.time.Clock()

        # COLORS
        self.BG_COLOR = (0, 0, 0)      # Black
        self.AGENT_COLOR = (0, 0, 255) # Blue
        self.GOAL_COLOR = (0, 255, 0)  # Green
        self.OBST_COLOR = (255, 0, 0)  # Red

    def draw(self, env):
        self.screen.fill(self.BG_COLOR)

        # Draw Goal
        gx1, gy1, gx2, gy2 = env.goal.get_bounding_box()
        pygame.draw.rect(self.screen, self.GOAL_COLOR, pygame.Rect(int(gx1), int(gy1), int(gx2 - gx1), int(gy2 - gy1)))
        
        # Draw Obstacles
        for obs in env.obstacles:
            ox1, oy1, ox2, oy2 = obs.get_bounding_box()
            pygame.draw.rect(self.screen, self.OBST_COLOR, pygame.Rect(int(ox1) int(oy1), int(ox2 - ox1), int(oy2 - oy1)))

        # Draw Agent
        ax, ay = env.agent_pos
        pygame.draw.circle(self.screen, self.AGENT_COLOR, (int(ax), int(ay)), 5)

        pygame.display.flip()
        # Apply Framerate Limit
        self.clock.tick(self.config.framerate)
