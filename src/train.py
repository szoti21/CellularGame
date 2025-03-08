import matplotlib.pyplot as plt
from game import GameEngine
from rlagent import Agent
from config import NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, NUM_OF_TRAININGS, SAVE_MODEL_AFTER_EVERY_X_TURNS


def train_agent():
    env = GameEngine()
    agent = Agent(state_dim=NUMBER_OF_INPUTS, action_dim=NUMBER_OF_OUTPUTS)
    rewards_per_episode = []

    for episode in range(NUM_OF_TRAININGS):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train()

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        agent.update_target_network()
        rewards_per_episode.append(total_reward)

        print(f"Episode {episode+1}: Reward: {total_reward:.2f}")

        if (episode + 1) % SAVE_MODEL_AFTER_EVERY_X_TURNS == 0:
            agent.save_model()

    # draw learning curve
    plt.plot(rewards_per_episode)
    plt.xlabel("#training")
    plt.ylabel("Total Reward")
    plt.show()

# Run training
train_agent()