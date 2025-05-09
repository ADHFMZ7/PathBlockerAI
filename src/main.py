import random

from env import Env
from config import GameSettings

def main():


    settings = GameSettings()

    env = Env(config: GameSettings)

    while True:
       
        actions = env.action_space

        # This is where we pick one of the actions to take
        action = random.choice(actions)

        obs, reward, terminated = env.step(action)

        if env.render():





if __name__ == "__main__":
    main()
