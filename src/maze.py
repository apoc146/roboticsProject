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
import pdb

sys.path.append(str(pathlib.Path(__file__).parent.parent))




## PARAMS
show_animation = True
canvasSize=35


## extrend class
class rrtStar(RRT):
	pass
	
def steps(source, dest):

	DISCRETIZATION_STEP=1
	dists = np.zeros(2, dtype=np.float32)
	for j in range(0,2):
		# print("\ndest state",dest.state)
		dists[j] = dest[j] - source[j]

	distTotal = magnitude(dists)
	#print(distTotal)
	curr = source
	path_steps = []
	if distTotal>0:
		incrementTotal = distTotal/DISCRETIZATION_STEP
		for j in range(0,2):
			dists[j] =dists[j]/incrementTotal

		numSegments = int(math.floor(incrementTotal))+1

		for i in range(0,numSegments):
			curr[0]=curr[0]+dists[0]
			curr[1]=curr[1]+dists[1]
			#print(curr)
			path_steps.append(list(curr))
		#pdb.set_trace()
	return path_steps


if __name__ == "__main__":
	
	print("Start " + __file__)
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
	env = gym.make(env_name, renders=True, wallDistractor=True, randomExplor=False, maxSteps=maxSteps, image_size=64, target_pos = (52.5,37.5),display_target=True)
	unit = 15
	obstacleList = [
	(0, 0,60,1), #left
	(0, 45, 60,1 ), #right
	(0.,0, 1,45), #top
	(15, 15, 45,1), #inleft
	(15, 30, 45,1), #inright
	(15, 15 , -1, 15), #bottom
	(60, 0, 1, 15), #bottom left
	( 60 ,30, -1, 15) #bottom right
	]
	"running env on 2 episodes"
	start = [unit / 2+ 3*unit, unit/2]
	goal = [unit/2 + 3*unit, unit*5/2]
	num_ep = 1
	rrt = RRT(start=start, goal=goal, randArea=[-canvasSize, canvasSize], obstacleList=obstacleList, dof=2, alg='rrtstar', geom='circle', maxIter=150)
	path = rrt.planning(animation=True)
	print("**********************")
	print('path',path)
	print("**********************\n")

	if(path==None):
		print("Path Not Found")
		sleep(2.5)
		exit(0)

	sleep(1.5)
	#path = [[52.5, 37.5], [28.756069441690805, 32.54737375510372], [14.838668738838471, 34.33536660701179], [11.220208816830862, 27.78248341420185], [9.334475311924216, 13.806401047017424], [21.21246209853887, 11.377836748374946], [33.600806395693695, 7.997243693299808], [52.5, 7.5]]
	#path = [start,[unit / 2+ 3*unit+0.5, unit/2] ,[unit / 2+ 3*unit+1, unit/2],[unit / 2+ 3*unit+2, unit/2], [unit / 2+ 3*unit+3, unit/2],[unit / 2+ 3*unit+4, unit/2]]
	#path =[ [52.5,37.5], [7.5,37.5],[7.5,7.5],[52.5,7.5]]

	# path = [[52.5, 37.5], [52.5, 37.5], [24.967109703537268, 35.713450804697914], [13.845633903763023, 35.8498902925696], [10.047764378458083, 34.374293364086924], [10.517495502554034, 7.824296555485333], [52.5, 7.5]]
	for episode in range(num_ep):
		obs_space = env.observation_space
		action_space = env.action_space
		
		obs = env.reset() 
		done = False
		
		for i in range(1,len(path)):
			path_steps = steps(path[i-1],path[i])
			for j in range(1,len(path_steps)):
				obs, reward, done, info = env.bot_step((np.array(path_steps[j])-np.array(path_steps[j-1]))/10,np.array(path_steps[j])/10)
				#print(obs)
				"get the image observation from the camera"
				sleep(0.5)
				obs = env.render(mode = "human")