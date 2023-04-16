import gym
from utils.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar

"register bullet_envs in gym"
import bullet_envs.__init__
import math
import sys
import matplotlib.pyplot as plt
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))




## PARAMS
show_animation = True


## extrend class
class rrtStar(RRTStar):
	def planning(self, animation=True):
		
		self.node_list = [self.start]
		for i in range(self.max_iter):
			print("Iter:", i, ", number of nodes:", len(self.node_list))
			rnd = self.get_random_node()
			nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
			new_node = self.steer(self.node_list[nearest_ind], rnd,
								self.expand_dis)
			near_node = self.node_list[nearest_ind]
			new_node.cost = near_node.cost + \
				math.hypot(new_node.x-near_node.x,
						new_node.y-near_node.y)

			if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
				near_inds = self.find_near_nodes(new_node)
				node_with_updated_parent = self.choose_parent(
					new_node, near_inds)
				if node_with_updated_parent:
					self.rewire(node_with_updated_parent, near_inds)
					self.node_list.append(node_with_updated_parent)
				else:
					self.node_list.append(new_node)

			if animation:
				self.draw_graph(rnd)

			if ((not self.search_until_max_iter)
					and new_node):  # if reaches goal
				last_index = self.search_best_goal_node()
				if last_index is not None:
					return self.generate_final_course(last_index)

		print("reached max iteration")

		last_index = self.search_best_goal_node()
		if last_index is not None:
			return self.generate_final_course(last_index)

		return None


if __name__ == "__main__":
	
	print("Start " + __file__)

	# ====Search Path with RRT====
	obstacle_list = [
		(5, 5, 1),
		(3, 6, 2),
		(3, 8, 2),
		(3, 10, 2),
		(7, 5, 2),
		(9, 5, 2),
		(8, 10, 1),
		(6, 12, 1),
	]  # [x,y,size(radius)]

	#Set Initial parameters
	rrt_star = rrtStar(
		start=[0, 0],
		goal=[6, 10],
		rand_area=[-2, 15],
		obstacle_list=obstacle_list,
		expand_dis=1,
		robot_radius=0.8)

	print("Started RRT* Path Planning\n")
	path = rrt_star.planning(animation=show_animation)
	print("Completed RRT* Path Planning\n")

	if path is None:
		print("Cannot find path")
	else:
		print("found path!!")

		# Draw final path
		if show_animation:
			rrt_star.draw_graph()
			plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
			plt.grid(True)
			plt.show()

	exit(0)

	## Turtle Bot ##

	'''
		The observation space: Box(-inf, inf, (3,), float32)
		The action space: Box(-1.0, 1.0, (2,), float32)
		Upper Bound for Env Observation [inf inf inf]
		Lower Bound for Env Observation [-inf -inf -inf]
		Upper Bound for Action [1. 1.]
		Lower Bound for Action [-1. -1.]

		print("The observation space: {}".format(obs_space))
		print("The action space: {}".format(action_space))
		print("Upper Bound for Env Observation", env.observation_space.high)
		print("Lower Bound for Env Observation", env.observation_space.low)
		print("Upper Bound for Action", env.action_space.high)
		print("Lower Bound for Action", env.action_space.low)
	'''

	env_name = 'TurtlebotMazeEnv-v0'
	actionRepeat = 1
	maxSteps = 100
	"OpenAI Gym env creation"
	env = gym.make(env_name, renders=True, wallDistractor=True, maxSteps=maxSteps, image_size=64, display_target=True)

	"running env on 2 episodes"
	num_ep = 2
	for episode in range(num_ep):
		obs_space = env.observation_space
		action_space = env.action_space

		obs = env.reset()
		done = False
		while not done:
			# follow a random policy
			action = env.action_space.sample()
			print(action)
			obs, reward, done, info = env.step(action)
			print(obs)
			"get the image observation from the camera"
			obs = env.render(mode = "human")