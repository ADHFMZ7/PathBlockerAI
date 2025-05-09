from dataclasses import dataclass, field
from pygame import Color

@dataclass
class GameSettings:
    screen_width: int = 400
    screen_height: int = 600
    player_speed: float = 5 
    fullscreen: bool = False
    framerate: int = 60
    bg_color: Color = field(default_factory=lambda : Color(0, 0, 0))
    render: bool = True
