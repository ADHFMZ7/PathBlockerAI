# PathBlocker

> [!WARNING]  
> We are still migrating PathBlocker from our old repository to this one. The code is unfinished.

PathBlocker is an interactive reinforcement learning project where an agent learns to navigate a dynamic, continuous 2D environment to reach a goal state while avoiding obstacles. The environment is rendered top-down and allows real-time user interaction, enabling users to modify the layout and behavior of the world.

This project explores how agents can adapt to unpredictable environments by using Deep Q-Learning, a deep reinforcement learning technique that enables agents to learn optimal behavior through trial and error.

## Features

- Deep Q-Learning Agent: The agent is trained using a neural network to approximate Q-values for different actions, enabling intelligent navigation in the maze.
- Interactive Environment: Built using Pygame, the environment supports real-time modifications. Users can move walls, add obstacles, and reposition the agent during training.
- Visualization Tools: Performance metrics and training progress are visualized using tools like Plotly to help track learning trends and agent behavior.

## Tech Stack

- Python – Core programming language
- Pygame – For building the interactive 2D environment
- PyTorch – For implementing and training the Deep Q-Learning model
- Plotly – For visualizing training metrics
- uv - Package management

## Getting Started

This project uses [`uv`](https://github.com/astral-sh/uv) for package management. To get started, started, follow these steps.

1. Clone repo
```bash
git clone https://github.com/ADHFMZ7/PathBlockerAI
cd PathBlockerAI
```
2. Install dependencies
```bash
uv sync
```

