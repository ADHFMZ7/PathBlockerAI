import torch
from agents.dqn import DQN
from env import Action

class DQNAgent:
    def __init__(self, model_path="trained_model.pth", n_observations=None, n_actions=None):
        """
        Initialize the DQN agent with a trained model
        
        Args:
            model_path: Path to the trained model weights
            n_observations: Number of observations in state space
            n_actions: Number of possible actions
        """
        self.model_path = model_path
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            self.model = DQN(self.n_observations, self.n_actions)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode
        
    def select_action(self, state):
        """
        Select the best action based on the current state
        
        Args:
            state: Current state observation
            
        Returns:
            Action: Selected action
        """
        with torch.no_grad():
            # Get action index with highest Q-value
            q_values = self.model(state)
            action_idx = torch.argmax(q_values).item()
            return Action[list(Action.__members__)[action_idx]]
            
    def __call__(self, observation):
        """
        Choose action based on observation
        
        Args:
            observation: Current state observation
            
        Returns:
            Action: Selected action
        """
        if self.model is None:
            self.load_model()
            
        return self.select_action(observation)