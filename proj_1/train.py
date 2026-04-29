from proj_1.flappy_bird import FlappyBirdEnv
from proj_1.agent import Agent
import numpy as np


def normalize(state):
    return state / np.array([500, 10, 400, 500])


def train():
    env = FlappyBirdEnv(render_mode=False)
    agent = Agent()

    episodes = 1000

    for ep in range(episodes):
        state = normalize(env.reset())
        total_reward = 0

        done = False

        while not done:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize(next_state)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        # epsilon decay
        agent.epsilon = max(0.05, agent.epsilon * 0.995)

        # target update
        if ep % 10 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        print(f"Episode {ep} | Reward: {total_reward} | Score: {env.score}")


if __name__ == "__main__":
    train()
