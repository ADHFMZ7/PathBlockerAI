import torch

from entities.obstacle import Obstacle

class Goal:

    def __init__(self, x: float, y: float, width: float = 15.0, height: float = 15.0):
        """
        Initialize the goal with its position and size.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
    def collides(self, other) -> bool:
        """
        Check if this goal collides with the given point.
        """

        if isinstance(other, Obstacle):
            # Check for collision with another obstacle
            return not (self.x + self.width < other.x or
                        self.x > other.x + other.width or
                        self.y + self.height < other.y or
                        self.y > other.y + other.height)
        # Check for collision with a point
        if isinstance(other, torch.Tensor): 
            x, y = other[0].item(), other[1].item()
            return (self.x <= x <= self.x + self.width) and (self.y <= y <= self.y + self.height)
    
    def get_position(self) -> tuple[float, float]:
        return self.x, self.y
    
    def get_bounding_box(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.x + self.width, self.y + self.height])
   
    def get_center(self) -> torch.tensor:
        return torch.tensor([self.x + self.width / 2, self.y + self.height / 2], dtype=torch.float32) 
        
