import gymnasium as gym

env = gym.make("Taxi-v3", render_mode="human")
env.reset()
env.render()

n_steps = 200
total_reward = 0
for i in range(n_steps):
    action = env.action_space.sample()
    print(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    env.render()
    if terminated or truncated:
        break

print("Total reward:", total_reward)
input()
env.close()
