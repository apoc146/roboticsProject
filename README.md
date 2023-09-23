# roboticsProject
RRT*-guided robot learning for planning

-       Setup maze environment with Turtlebot in Pybullet
-       Run RRT* to gather data, including local lidar observations
-       Implement an imitation learning policy to generate the next waypoint toward the goal using only the local lidar observations

You can find the results and documentation in the [REPORT](./CS593Report.pdf)

## Instructions
- Run python3 mazeV.py for checkpoint 3 
Checkpoint 3 includes code for Collection of RRT data, Behaviour Cloning without PPO expert, DAgger with PPO expert
- Run python3 maze.py for checkpoint 2

- Conda Env : condaEnv.txt
Project files :
 - maze.py - Checkpount 1 & 2
 - mazeV.py - Checkpoint 3
 - turtleEnv.py - Containes the turtlebot environment along with the 2 maze environemnts with robot 
 - dagger.py - sample code for runing one episode of dagger
 - bc.py - sample code for running one episode of behaviour cloning
 - path.docx - file for storing RRT paths
 - transitions.pkl - contains the observations (LIDAR, Position(included only for maze 1)) and actions space
 - RRTStar.py - code for implementing RRT*


WIP : Imitation Learning
