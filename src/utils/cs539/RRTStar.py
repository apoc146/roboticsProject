"""
Path Planning Sample Code with RRT*

author: Ahmed Qureshi, code adapted from AtsushiSakai(@Atsushi_twi)

"""


import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

## --- PARAMS --- ##
#Q5 global radius 1
radius=2.5
canvasSize=100
mazePolygon=[(0,0),(0,45),(60,45),(60,30),(15,30),(15,15),(60,15),(60,0)]

##Q6 global params
rectRoboLength=1.5
rectRoboWidth=3
rectRoboTheta=0
pi=np.pi


##My functions

def isPointInside(point,polygonPoints):
	pt=Point(point)
	polygon=Polygon(polygonPoints)
	return polygon.contains(pt)

def ballRegionVal(gamma,vertexCount,dims):
	return gamma*(math.log(vertexCount)/vertexCount)**(1.0/dims)


def thetaSample():
	return random.uniform(-2*pi,2*pi)

#### Given Code Below ####

def diff(v1, v2):
	"""
	Computes the difference v1 - v2, assuming v1 and v2 are both vectors
	"""
	return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
	"""
	Computes the magnitude of the vector v.
	"""
	return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
	"""
	Computes the Euclidean distance (L2 norm) between two points p1 and p2
	"""
	return magnitude(diff(p1, p2))

class RRT():
	"""
	Class for RRT Planning
	"""

	def __init__(self, start, goal, obstacleList, randArea, alg, geom, dof=2, expandDis=0.1, goalSampleRate=5, maxIter=500):
		"""
		Sets algorithm parameters

		start:Start Position [x,y]
		goal:Goal Position [x,y]
		obstacleList:obstacle Positions [[x,y,width,height],...]
		randArea:Ramdosm Samping Area [min,max]

		"""
		self.start = Node(start)
		self.end = Node(goal)
		if(geom=="rectangle"):
			self.start.state.append(0)
			self.end.state.append(0)

		self.obstacleList = obstacleList
		self.minrand = randArea[0]
		self.maxrand = randArea[1]
		self.alg = alg
		self.geom = geom
		self.dof = dof

		self.expandDis = expandDis
		self.goalSampleRate = goalSampleRate
		self.maxIter = maxIter

		self.goalfound = False
		self.solutionSet = set()

	def planning(self, animation=False):
		"""
		Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
		You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

		animation: flag for animation on or off
		"""

		self.nodeList = [self.start]
		for i in range(self.maxIter):
		
			path = self.get_path_to_goal()
			if not path:
				rnd = self.generatesample()
				while(isPointInside((rnd.state[0],rnd.state[1]),mazePolygon)==False):
					rnd=self.generatesample()

				nind = self.GetNearestListIndex(self.nodeList, rnd)
				rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])


				if (rnd_valid):
					newNode = copy.deepcopy(rnd)
					newNode.parent = nind
					newNode.cost = rnd_cost + self.nodeList[nind].cost

					if self.alg == 'rrtstar':
						nearinds = self.find_near_nodes(newNode) # you'll implement this method
						# print("\nabc")
						# print(len(nearinds))
						# print("\ndef")
						newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
					else:
						newParent = None

					# insert newNode into the tree
					if newParent is not None:
						newNode.parent = newParent
						newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
					else:
						pass # nind is already set as newNode's parent
					self.nodeList.append(newNode)
					newNodeIndex = len(self.nodeList) - 1
					self.nodeList[newNode.parent].children.add(newNodeIndex)

					if self.alg == 'rrtstar':
						self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

					if self.is_near_goal(newNode):
						self.solutionSet.add(newNodeIndex)
						self.goalfound = True

					if animation:
						self.draw_graph(rnd.state)
						time.sleep(5)
			else:
				break
					

		return path


		

	def choose_parent(self, newNode, nearinds):
		"""
		Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

		newNode: the node to be inserted
		nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

		Returns: index of the new parent selected
		"""
		# your code here
		if len(nearinds)==0:
			return None
		min_cost=999999999999
		for idx in nearinds:
			node=self.nodeList[idx]
			is_obstacle_free, new_cost = self.steerTo(newNode, node)
			if is_obstacle_free:
				if (node.cost+new_cost<min_cost):
					min_cost=node.cost+new_cost
					min_node=idx
		return min_node


	def steerTo(self, dest, source):
		"""
		Charts a route from source to dest, and checks whether the route is collision-free.
		Discretizes the route into small steps, and checks for a collision at each step.

		This function is used in planning() to filter out invalid random samples. You may also find it useful
		for implementing the functions in question 1.

		dest: destination node
		source: source node

		returns: (success, cost) tuple
			- success is True if the route is collision free; False otherwise.
			- cost is the distance from source to dest, if the route is collision free; or None otherwise.
		"""

		newNode = copy.deepcopy(source)

		DISCRETIZATION_STEP=self.expandDis

		dists = np.zeros(self.dof, dtype=np.float32)
		for j in range(0,self.dof):
			# print("\ndest state",dest.state)
			dists[j] = dest.state[j] - source.state[j]

		distTotal = magnitude(dists)


		if distTotal>0:
			incrementTotal = distTotal/DISCRETIZATION_STEP
			for j in range(0,self.dof):
				dists[j] =dists[j]/incrementTotal

			numSegments = int(math.floor(incrementTotal))+1

			stateCurr = np.zeros(self.dof,dtype=np.float32)
			for j in range(0,self.dof):
				stateCurr[j] = newNode.state[j]

			stateCurr = Node(stateCurr)

			for i in range(0,numSegments):

				if not self.__CollisionCheck(stateCurr):
					return (False, None)

				for j in range(0,self.dof):
					stateCurr.state[j] += dists[j]

			if not self.__CollisionCheck(dest):
				return (False, None)

			return (True, distTotal)
		else:
			return (False, None)

	def generatesample(self):
		"""
		Randomly generates a sample, to be used as a new node.
		This sample may be invalid - if so, call generatesample() again.

		You will need to modify this function for question 3 (if self.geom == 'rectangle')

		returns: random c-space vector
		"""
		if random.randint(0, 100) > self.goalSampleRate:
			sample=[]
			for j in range(0,self.dof):
				## for rectangle -> the third param 'Theta' doesnt use the same range [minrand,maxrand] for sampling
				if(j==2 and (self.geom=="rectangle")):
					sample.append(thetaSample())
				else:
					sample.append(random.uniform(self.minrand, self.maxrand))
			rnd=Node(sample)
		else:
			rnd = self.end
		return rnd

	def is_near_goal(self, node):
		"""
		node: the location to check

		Returns: True if node is within 5 units of the goal state; False otherwise
		"""
		d = dist(node.state, self.end.state)
		if d < 5.0:
			return True
		return False

	@staticmethod
	def get_path_len(path):
		"""
		path: a list of coordinates

		Returns: total length of the path
		"""
		pathLen = 0
		for i in range(1, len(path)):
			pathLen += dist(path[i], path[i-1])

		return pathLen


	def gen_final_course(self, goalind):
		"""
		Traverses up the tree to find the path from start to goal

		goalind: index of the goal node

		Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
		"""
		path = [self.end.state]
		while self.nodeList[goalind].parent is not None:
			node = self.nodeList[goalind]
			path.append(node.state)
			goalind = node.parent
		path.append(self.start.state)
		return path

	def find_near_nodes(self, newNode):
		"""
		Finds all nodes in the tree that are "near" newNode.
		See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

		newNode: the node to be inserted.

		Returns: a list of indices of nearby nodes.
		"""

		# Use this value of gamma
		GAMMA = 50
		near_nodes=[]
		vertexCount=len(self.nodeList)
		dims=2
		radiusRange=ballRegionVal(GAMMA,vertexCount,dims)

		for listNode in self.nodeList:
			## If there is a collision path -> skip this node as near node
			if(self.steerTo(newNode,listNode)[0]==False):
				continue
			if dist(listNode.state,newNode.state)< radiusRange:
				near_nodes.append(self.nodeList.index(listNode))
		# your code here
		return near_nodes


	## O(exponential) solution - less efficient
	## updated cost for all children
	def updateChildNodeCost(self,parentNode):
		for node in self.nodeList:
			##If child of parentNode
			if node.parent==parentNode:
				node.cost=parentNode.cost+dist(parentNode.state,node.state)
				self.updateChildNodeCost(node)



	def rewire(self, newNode, newNodeIndex, nearinds):
		"""
		Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
		Recall that a node should be rewired if doing so would reduce its cost.

		newNode: the node that was just inserted
		newNodeIndex: the index of newNode
		nearinds: list of indices of nodes near newNode
		"""
		nodeList=self.nodeList

		## If no near points then dont rewire
		if(nearinds==None or nearinds==[]):
			return

		for idx in nearinds:
			node=nodeList[idx]
			## if No parent -> start node -> then skip
			if(node.parent==None):
				continue

			parentOfNodeIdx=node.parent
			parentOfNode=self.nodeList[parentOfNodeIdx]

			currentCost=node.cost
			newNodeCost=newNode.cost
			costWithNewNodeParent=newNodeCost+dist(newNode.state,node.state)

			##Skip if there is a collision -> cant rewire
			if(self.steerTo(newNode,node)[0]==False):
				continue


			if(costWithNewNodeParent<currentCost):
				# print("\n---Rewire Happened Here---\n")
				##change parents
				#break old parent-Remove node as child from its parent
				parentOfNode.children.remove(idx)

				#make new parent
				node.parent=newNodeIndex
				
				#update cost
				node.cost=costWithNewNodeParent
				
				#add as child to parent
				newNode.children.add(idx)

				##Recursively update newNode's children's cost
				self.updateChildNodeCost(newNode)

	def GetNearestListIndex(self, nodeList, rnd):
		"""
		Searches nodeList for the closest vertex to rnd

		nodeList: list of all nodes currently in the tree
		rnd: node to be added (not currently in the tree)

		Returns: index of nearest node
		"""
		dlist = []
		for node in nodeList:
			dlist.append(dist(rnd.state, node.state))

		minind = dlist.index(min(dlist))

		return minind

	## 3rd collision check
	## CIRCLE/RECTANGLE
	def circleRect(self, cx,  cy,  radius,  rx,  ry,  rw,  rh):

		# // temporary variables to set edges for testing
		testX = cx
		testY = cy

		# // which edge is closest
		if (cx < rx):
			testX = rx      
		elif(cx > rx+rw):
			testX = rx+rw
		if (cy < ry):
			testY = ry    
		elif(cy > ry+rh):
			testY = ry+rh

		# // get distance from closest edges
		distX = cx-testX
		distY = cy-testY
		distance = math.sqrt( (distX*distX) + (distY*distY) )

		# // if the distance is less than the radius, collision!
		if (distance <= radius):
			return True
		
		return True


	## 2nd collision checker 
	# collision - Return True
	def intersects(self, cx,  cy,  radius,  left,  right, bottom, top):
	
		closestX=0
		closestY=0
		if(cx<left):
			closestX=left
		else:
			if cx>right:
				closestX=right
			else:
				closestX=cx


		if(cy<top):
			closestY=top
		else:
			if cy>bottom:
				closestY=bottom
			else:
				closestY=cy

   		# closestX = (cx < left ? left : (cx > right ? right : cx));
   		# closestY = (cy < top ? top : (cy > bottom ? bottom : cy));

		dx = closestX - cx
		dy = closestY - cy

		return ( dx * dx + dy * dy ) <= radius * radius
	


	def circleCollisionCheck(self,rectLeftX,rectBottomY,wd,ht,circleCenterX,circleCenterY,r):
		## lets find boundary range of rectangle
		rectRigthX=rectLeftX+wd
		rectTopY=rectBottomY+ht

		## circle boundary
		circleLeft=circleCenterX-r
		circleRight=circleCenterX+r
		circleBottom=circleCenterY-r
		circleTop=circleCenterY+r

		#Lets do bounding simple box checks - Here no intersection
		if(circleRight < rectLeftX or circleLeft > rectRigthX or rectBottomY > circleTop or rectTopY < circleBottom):
			return False

		## check if circle is inside rectangle
		# for i in range(int(rectLeftX),int(rectRigthX)):
		# 	for j in range(int(rectBottomY),int(rectTopY)):
		# 		if( ((i-circleCenterX)**2+(j-circleCenterY)**2-r**2)<=0):
		# 			print("INSIDE AREA CHECK\n")
		# 			return True

		for i in  np.arange(rectLeftX,rectRigthX,0.1) :
			for j in np.arange(rectBottomY,rectTopY,0.1):
				if( (math.sqrt((i-circleCenterX)**2+(j-circleCenterY)**2)-r)<0):

					## Uncomment for Logs
					# print("******* COLLISION DETECTED OPEN*********\n")
					# print("\nINSIDE AREA CHECK\n")
					# print("i:",i)
					# #print("\n")
					# print("j:",j)
					# #print("\n")
					# print("cx:",circleCenterX)
					# #print("\n")
					# print("cy:",circleCenterY)
					# print("dist below:",math.sqrt((i-circleCenterX)**2+(j-circleCenterY)**2)-r)
					# #print("\n")
					# print("\nOutside AREA CHECK\n")
					# print("******* COLLISION DETECTED CLOSE*********\n")

					return True
		
		#check center inside
		if((rectLeftX<= circleCenterX <=rectRigthX) and (rectBottomY<=circleCenterY<=rectTopY)):
			return True
		
		return False

	def circleCheckUtil(self,ox,oy,w,h,cx,cy,r):
		return self.circleCollisionCheck(ox,oy,w,h,cx,cy,r)

		##def circleRect(self, cx,  cy,  radius,  rx,  ry,  rw,  rh):

		## def intersects(self, cx,  cy,  radius,  left,  right, bottom, top):

		# return self.intersects(cx,cy,r,ox,ox+w,oy,oy+h)
	
	def rectCheckUtil(self,rCenterX,rCenterY,theta,w,h,ox,oy,ow,oh):
		return self.rectCollisionCheck(rCenterX,rCenterY,theta,w,h,ox,oy,ow,oh)

	def check_overlap(self,line1, line2):
		x1_start, x1_end = line1
		x2_start, x2_end = line2
		if x1_end < x2_start or x2_end < x1_start:
			return False
		return True


	## Collision checker for a rectangle
	def rectCollisionCheck(self,x,y,theta,w,h,ox,oy,ow,oh):

		centerX=x
		centerY=y

		## find projections for all points
		## for X points

		bottomLeftX=centerX-((w/2.0)*math.cos(theta)+(h/2)*math.sin(theta))
		bottomLeftY=centerY-((w/2.0)*math.sin(theta)-(h/2)*math.cos(theta))
		bottomLeft=[bottomLeftX,bottomLeftY]

		bottomRightX=centerX+((w/2.0)*math.cos(theta)+(h/2)*math.sin(theta))
		bottomRightY=centerY+((w/2.0)*math.sin(theta)-(h/2)*math.cos(theta))
		bottomRight=[bottomRightX,bottomRightY]

		topRightX=centerX+((w/2.0)*math.cos(theta)-(h/2)*math.sin(theta))
		topRightY=centerY+((w/2.0)*math.sin(theta)+(h/2)*math.cos(theta))
		topRight=[topRightX,topRightY]

		topLeftX=centerX-((w/2.0)*math.cos(theta)-(h/2)*math.sin(theta))
		topLeftY=centerY-((w/2.0)*math.sin(theta)+(h/2)*math.cos(theta))
		topLeft=[topLeftX,topLeftY]

		rectanglePoints = [bottomLeft,bottomRight, topRight, topLeft]

		## Obstacle projection
		oBottomLeftX=ox
		oBottomLeftY=oy

		oBottomRightX=ox+ow
		oBottomRightY=oy

		oTopRightX=ox+ow
		oTopRightY=oy+oh
		
		oTopLeftX=ox
		oTopLeftY=oy+oh

		## Rect Projection range 
		xMax=max([rectanglePoints[0][0],rectanglePoints[1][0],rectanglePoints[2][0],rectanglePoints[3][0]])
		xMin=min([rectanglePoints[0][0],rectanglePoints[1][0],rectanglePoints[2][0],rectanglePoints[3][0]])

		yMax=max([rectanglePoints[0][1],rectanglePoints[1][1],rectanglePoints[2][1],rectanglePoints[3][1]])
		yMin=min([rectanglePoints[0][1],rectanglePoints[1][1],rectanglePoints[2][1],rectanglePoints[3][1]])


		## Obs Projection range
		oXMax=max([oBottomLeftX,oBottomRightX,oTopRightX,oTopLeftX])
		oXMin=min([oBottomLeftX,oBottomRightX,oTopRightX,oTopLeftX])

		oYMax=max([oBottomLeftY,oBottomRightY,oTopRightY,oTopLeftY])
		oYMin=min([oBottomLeftY,oBottomRightY,oTopRightY,oTopLeftY])

		## check intersection on X axis
		isRectCollisionX=self.check_overlap([xMin,xMax],[oXMin,oXMax])
		isRectCollisionY=self.check_overlap([yMin,yMax],[oYMin,oYMax])

		if(isRectCollisionX and isRectCollisionY):
			return True
		return False




	def __CollisionCheck(self, node):
		"""
		Checks whether a given configuration is valid. (collides with obstacles)

		You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
		"""
		if not isPointInside((node.state[0],node.state[1]),mazePolygon):
			return False
		### For rectangle-Returns If collision 
		if(self.geom=="rectangle"):
			for (ox, oy, sizex,sizey) in self.obstacleList:
				obs=[ox+sizex/2.0,oy+sizey/2.0]
				obs_size=[sizex,sizey]
				cf = False
				rCenterX=node.state[0]
				rCenterY=node.state[1]
				theta=node.state[2]
				#Declare As GLOBAL at top on file
				# rectRoboLength=1.5
				# rectRoboWidth=3
				isRectCollision=self.rectCheckUtil(rCenterX,rCenterY,theta,rectRoboWidth,rectRoboLength,ox,oy,sizex,sizey)
				if(isRectCollision):
					return False
			return True


		s = np.zeros(2, dtype=np.float32)
		s[0] = node.state[0]
		s[1] = node.state[1]


		# if(self.geom=="circle"):
		# 	for (ox, oy, sizex,sizey) in self.obstacleList:
		# 		if(self.circleCollisionCheck(ox,oy,sizex,sizey,node.state[0],node.state[1],1)==True):
		# 			return False

		# 		# if(self.intersects(node.state[0],node.state[1],1,ox,ox+sizex,oy,oy+sizey)==True):
		# 		# 	return True

		# 		# if(self.circleRect(node.state[0],node.state[1],1,ox,oy,sizex,sizey)==True):
		# 		# 	return False
		# 	return True

		for (ox, oy, sizex,sizey) in self.obstacleList:
			obs=[ox+sizex/2.0,oy+sizey/2.0]
			obs_size=[sizex,sizey]
			cf = False
			for j in range(self.dof):
				if abs(obs[j] - s[j])>obs_size[j]/2.0:
					cf=True

					## check for circle now
					if(self.geom=="circle"):
						isCollision = self.circleCheckUtil(ox,oy,sizex,sizey,node.state[0],node.state[1],radius)
						cf=not isCollision
						if(cf==False):
							# print("\n-----Marked Collision\n")
							return cf
					break
			if cf == False:
				return False

		return True  # safe'''

	def get_path_to_goal(self):
		"""
		Traverses the tree to chart a path between the start state and the goal state.
		There may be multiple paths already discovered - if so, this returns the shortest one

		Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
		"""
		if self.goalfound:
			goalind = None
			mincost = float('inf')
			for idx in self.solutionSet:
				cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state)
				if goalind is None or cost < mincost:
					goalind = idx
					mincost = cost
			return self.gen_final_course(goalind)
		else:
			return None

	def plot_circle(self,center_x, center_y, radius):
		angle = np.linspace(0, 2*np.pi, 100)
		x = center_x + radius * np.cos(angle)
		y = center_y + radius * np.sin(angle)
		plt.plot(x, y)
		# plt.axis('equal')
		# plt.show()

	def draw_graph(self, rnd=None):
		"""
		Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

		You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
		"""
		plt.clf()
		# for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect(
			'key_release_event',
			lambda event: [exit(0) if event.key == 'escape' else None])

		for (ox, oy, sizex, sizey) in self.obstacleList:
			rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
			plt.gca().add_patch(rect)

		set_visible_circle=False
		set_visible_rec=False

		for node in self.nodeList:
			if node.parent is not None:
				if node.state is not None:
					if self.geom == 'circle':
						if set_visible_circle==True:
							circle.set_visible(False)

						circle = mpatches.Circle((node.state[0], node.state[1]), radius ,fill=True, color="yellow")
						plt.gca().add_patch(circle)
						set_visible_circle = True

					plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
						node.state[1], self.nodeList[node.parent].state[1]], "-g")
					
					if self.geom=="rectangle":
						if set_visible_rec:
							rect.set_visible(False)

						theta = (node.state[2]*180)/pi
						rect = mpatches.Rectangle(xy = (node.state[0]-1.5, node.state[1]-0.75), width=3, height=1.5, angle=theta, rotation_point='center', fill=True, color="black", linewidth=0.1)
						plt.gca().add_patch(rect)

						

		if self.goalfound:
			path = self.get_path_to_goal()
			x = [p[0] for p in path]
			y = [p[1] for p in path]
			if self.geom=="circle":
				plt.plot(x, y, 'bo', linewidth=2, markersize=10)
			
			if self.geom=="rectangle":
				plt.plot(x, y, 'bs', linewidth=2, markersize=10)


			plt.plot(x, y, '-r')

		if rnd is not None:
			if(self.geom=="circle"):
				self.plot_circle(node.state[0],node.state[1],radius)

			plt.plot(rnd[0], rnd[1], "^k")

		plt.plot(self.start.state[0], self.start.state[1], "xr")
		plt.plot(self.end.state[0], self.end.state[1], "xr")
		plt.axis("equal")
		plt.axis([-canvasSize, canvasSize, -canvasSize, canvasSize])
		plt.grid(True)
		plt.pause(0.01)



class Node():
	"""
	RRT Node
	"""

	def __init__(self,state):
		self.state =state
		self.cost = 0.0
		self.parent = None
		self.children = set()



def main():
	parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
	parser.add_argument('-g', '--geom', default='circle', choices=['point', 'circle', 'rectangle'], \
		help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
	parser.add_argument('--alg', default='rrtstar', choices=['rrt', 'rrtstar'], \
		help='which path-finding algorithm to use. default: "rrt"')
	parser.add_argument('--iter', default=150, type=int, help='number of iterations to run')
	parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
	parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')

	args = parser.parse_args()

	show_animation = not args.blind and not args.fast

	print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
	starttime = time.time()


	obstacleList = [
	(-15,0, 15.0, 5.0),
	(15,-10, 5.0, 10.0),
	(-10,8, 5.0, 15.0),
	(3,15, 10.0, 5.0),
	(-10,-10, 10.0, 5.0),
	(5,-5, 5.0, 5.0),
	]

	start = [-10, -17]
	goal = [10, 10]
	dof=2
	if(args.geom == "rectangle" and dof!=3):
		print("\n\t-*-*-*-*-*-*- Rectangle Body Should have DOF=3 -> X,Y,Theta. Please Correct -*-*-*-*-*-*-\n\n")
		print("\t-*-*-*-*-*-*- Setting DOF to 3 -*-*-*-*-*-*-\n\n")
		dof=3
		
	print()

	rrt = RRT(start=start, goal=goal, randArea=[-20, 20], obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter)
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


if __name__ == '__main__':
	main()


