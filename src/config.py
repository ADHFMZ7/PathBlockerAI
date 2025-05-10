from dataclasses import dataclass, field
from pygame import Color

@dataclass
class GameSettings:
    screen_width: int = 600
    screen_height: int = 600
    agent_speed: float = 5 
    fullscreen: bool = False
    framerate: int = 60
    bg_color: Color = field(default_factory=lambda : Color(0, 0, 0))
    render: bool = True
    num_obstacles: int = 15

@dataclass
class TrainSettings:
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000
    target_update: int = 10
    learning_rate: float = 0.001
    memory_size: int = 10000
    num_episodes: int = 500
    max_steps_per_episode: int = 500
    game_settings: GameSettings = field(default_factory=GameSettings)

