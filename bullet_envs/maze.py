import gym

"register bullet_envs in gym"
import bullet_envs.__init__

env_name = 'TurtlebotMazeEnv-v0'
actionRepeat = 1
maxSteps = 100
"OpenAI Gym env creation"
env = gym.make(env_name, renders=True, wallDistractor=True, maxSteps=maxSteps, image_size=64, display_target=True)

"running env on 2 episodes"
num_ep = 2
for episode in range(num_ep):
    obs = env.reset()
    done = False
    while not done:
        # follow a random policy
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        "get the image observation from the camera"
        obs = env.render()