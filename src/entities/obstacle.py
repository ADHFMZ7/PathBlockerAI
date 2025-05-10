import torch

class Obstacle:
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @staticmethod
    def random(width, height, size_scale=0.25) -> "Obstacle":
        """
        Generate a random obstacle within the given width and height.
        """
        x = torch.randint(0, width - 1, (1,)).item()
        y = torch.randint(0, height - 1, (1,)).item()
        obstacle_width = int(torch.randint(20, int(width * size_scale), (1,)).item())
        obstacle_height = int(torch.randint(20, int(height * size_scale), (1,)).item())
        return Obstacle(x, y, obstacle_width, obstacle_height)

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def get_size(self) -> tuple[float, float]:
        return self.width, self.height
    
    def get_bounding_box(self) -> torch.Tensor:
        return torch.tensor([self.x, self.y, self.x + self.width, self.y + self.height])
    
    def collides(self, other) -> bool:
        """
        Check if this obstacle collides with the given point or obstacle.
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
