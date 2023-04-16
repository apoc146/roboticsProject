import gym
# from utils.PythonRobotics.PathPlanning.RRTStar.rrt_star import RRTStar
from utils.cs539.RRTStar import *

"register bullet_envs in gym"
import bullet_envs.__init__
import math
import sys
import matplotlib.pyplot as plt
import pathlib
from time import sleep


sys.path.append(str(pathlib.Path(__file__).parent.parent))




## PARAMS
show_animation = True
canvasSize=35


## extrend class
class rrtStar(RRT):
	pass
	


if __name__ == "__main__":
	
	print("Start " + __file__)
	

	parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
	parser.add_argument('-g', '--geom', default='circle', choices=['point', 'circle', 'rectangle'], \
		help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
	parser.add_argument('--alg', default='rrtstar', choices=['rrt', 'rrtstar'], \
		help='which path-finding algorithm to use. default: "rrt"')
	parser.add_argument('--iter', default=150, type=int, help='number of iterations to run')
	parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
	parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')

	show_animation=True

	args = parser.parse_args()

	show_animation = not args.blind and not args.fast

	print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
	starttime = time.time()


	obstacleList = [
	##maze start
	(-30,20, 70, 2.0),
	(40,20, 2.0, -40.0),
	(40,-20, -70, -2.0),
	(-30,-20, 2, 15.0),
	(-30,-20+15, 25, 2.0),
	(-30+25,-20+15, 2, 10),
	(-30,-20+15+10, 25, -2.0),
	(-30,-20+15+10, 2, 15.0),
	## maze end
	
	## block
	(5,-5, 5.0, 5.0),
	]

	start = [-10, -17]
	goal = [-20, 10]
	dof=2
	if(args.geom == "rectangle" and dof!=3):
		print("\n\t-*-*-*-*-*-*- Rectangle Body Should have DOF=3 -> X,Y,Theta. Please Correct -*-*-*-*-*-*-\n\n")
		print("\t-*-*-*-*-*-*- Setting DOF to 3 -*-*-*-*-*-*-\n\n")
		dof=3
		
	print()

	rrt = RRT(start=start, goal=goal, randArea=[-canvasSize, canvasSize], obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter)
	path = rrt.planning(animation=show_animation)

	endtime = time.time()

	if path is None:
		print("FAILED to find a path in %.2fsec"%(endtime - starttime))
	else:
		print("SUCCESS - found path of cost %.5f in %.2fsec"%(RRT.get_path_len(path), endtime - starttime))
	# Draw final path
	if not args.blind:
		rrt.draw_graph()
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