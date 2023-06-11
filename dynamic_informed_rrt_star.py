
# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import random


# function to animate/ visualize the node expoloration and the generated path
def animate(viz, pathTaken, start_pos, goal_pos, path, dynamicPos = [-1,-1]):
    imCtr = 0
    plt.rcParams["figure.figsize"] = [30,20]
    fig, ax = plt.subplots()
    ax.axis('on')
    ax.margins(1)
    plt.xlim(0,300)
    plt.ylim(0,200)

    # set goal and start
    ax.scatter(start_pos[0],start_pos[1],color = "red")
    ax.scatter(goal_pos[0],goal_pos[1],color = "green")

    # draw obstacle space
    xObs, yObs = np.meshgrid(np.arange(0, 320), np.arange(0, 200))
    square1 = plt.Rectangle((70, 147.5), 20, 20, fc='black')
    ax.add_artist(square1)

    square2 = plt.Rectangle((70, 92.5), 20, 20, fc='black')
    ax.add_artist(square2)

    square3 = plt.Rectangle((70, 37.5), 20, 20, fc='black')
    ax.add_artist(square3)

    square4 = plt.Rectangle((145, 185), 20, 20, fc='black')
    ax.add_artist(square4)

    square5 = plt.Rectangle((145, 125), 20, 20, fc='black')
    ax.add_artist(square5)

    square6 = plt.Rectangle((145, 60), 20, 20, fc='black')
    ax.add_artist(square6)

    square7 = plt.Rectangle((145, 0), 20, 20, fc='black')
    ax.add_artist(square7)

    square8 = plt.Rectangle((220, 147.5), 20, 20, fc='black')
    ax.add_artist(square8)

    square9 = plt.Rectangle((220, 92.5), 20, 20, fc='black')
    ax.add_artist(square9)

    square10 = plt.Rectangle((220, 37.5), 20, 20, fc='black')
    ax.add_artist(square10)

    ax.plot([300,300],[0,200], linestyle='-', color="red")

    boundary1 = (xObs<=1) 
    ax.fill(xObs[boundary1], yObs[boundary1], color='black')
    boundary2 = (xObs>=319) 
    ax.fill(xObs[boundary2], yObs[boundary2], color='black')
    boundary3 = (yObs<=1) 
    ax.fill(xObs[boundary3], yObs[boundary3], color='black')
    boundary4 = (yObs>=199) 
    ax.fill(xObs[boundary4], yObs[boundary4], color='black')
    ax.set_aspect(1)
    
    start_time = time.time()
    startX = []
    startY = []
    endX = []
    endY = []
    explored_startX = []
    explored_startY = []
    explored_endX = []
    explored_endY = []
    count = 0

    for index in range(1, len(viz)):
        parentNode = path[viz[index]]
        explored_startX.append(parentNode[0])
        explored_startY.append(parentNode[1])
        explored_endX.append(viz[index][0] - parentNode[0])
        explored_endY.append(viz[index][1] - parentNode[1])    
        count = count + 1

    # backtrack space
    if(len(pathTaken) > 0):
        for index in range(1, len(pathTaken)):
            startX.append(pathTaken[index-1][0])
            startY.append(pathTaken[index-1][1])
            endX.append(pathTaken[index][0] - pathTaken[index-1][0])
            endY.append(pathTaken[index][1] - pathTaken[index-1][1])    
            
            count = count + 1
    plt.quiver(np.array((explored_startX)), np.array((explored_startY)), np.array((explored_endX)), np.array((explored_endY)), units = 'xy', scale = 1, color = 'cyan', label = 'Explored region', headwidth= 1, headlength= 2)
    if dynamicPos != [-1,-1]:
        cc = plt.Circle(( dynamicPos[0] , dynamicPos[1] ), 10, color = "black") 
        ax.add_artist( cc )

    if(len(pathTaken) > 0):
        plt.quiver(np.array((startX)), np.array((startY)), np.array((endX)), np.array((endY)), units = 'xy', scale = 1, color = 'r', label = 'Backtrack path', headwidth= 1, headlength= 2)
    plt.legend()
    plt.show()
    plt.close()
    end_time = time.time()
    print('Time taken to visualize: ',(end_time - start_time)," sec")

# function to check if point is in map limits
def IsValid(currX, currY, gap=10):

    xMax, yMax = [300, 200]
    xMin, yMin = [0, 0]

    if currY <= gap or currY >= yMax- gap or currX <= gap or currX >= xMax- gap:
        return False
    else:
        return True

# function to check in the point is in obastacle space
def IsObstacle(x, y,gap=10, dynamicPos = [-1,-1]):

    # constants
    xMax, yMax = [300, 200]
    xMin, yMin = [0, 0]
    if dynamicPos != [-1,-1]:
        if (x-dynamicPos[0])**2 + (y-dynamicPos[1])**2 <= (10+3)**2:
            return True
    # square 1
    if x>= 70-gap and x<= 85+gap and y >=147.5-gap and y <= 162.5+gap:
        return True

    # square 2
    elif x>= 70-gap and x<= 85+gap and y >=92.5-gap  and y <= 109.5+gap:
        return True

    # square 3
    elif x>= 70-gap and x<= 85+gap and y >=37.5-gap  and y <= 52.5+gap:
        return True

    # square 4
    elif x>= 145-gap and x<= 160+gap and y >=185-gap  and y <= yMax:
        return True

    # square 5
    elif x>= 145-gap and x<= 160+gap and y >=125-gap  and y <= 140+gap:
        return True

    # square 6
    elif x>= 145-gap and x<= 160+gap and y >=60-gap  and y <= 75+gap:
        return True

    # square 7
    elif x>= 145-gap and x<= 160+gap and y >=yMin  and y <= 20+gap:
        return True

    # square 8
    elif x>= 220-gap and x<= 235+gap and y >=147.5-gap  and y <= 162.5+gap:
        return True

    # square 9
    elif x>= 220-gap and x<= 235+gap and y >=92.5-gap  and y <= 109.5+gap:
        return True

    # square 10
    elif x>= 220-gap and x<= 235+gap and y >=37.5-gap  and y <= 52.5+gap:
        return True

    else:
        return False

# function to select a random position in the planned path 
# used to spawn the dynamic obstacle
def generateRandomObstacle(backtrackStates):
    i = round(0.2*len(backtrackStates))
    j = round(0.8*len(backtrackStates))
    return random.choice(backtrackStates[i:j])

# function to check if any of the waypoints in generated path lie in the new obstacle space
# return a list of waypoints lying in the new obstacle space
def validatePath(backtrackStates,newObsPos):
    invalidNodes = []
    flag = False
    for x,y in backtrackStates:
        if IsObstacle(x,y,dynamicPos = newObsPos):
            invalidNodes.append((x,y))
            flag = True
    return flag,invalidNodes

# function to prune the tree and disconnect invalid nodes
def disconnectInvalidNodes(invalidNodes,backtrackStates,childParentRelation,c2cInfo):

    formerIdx = backtrackStates.index(invalidNodes[0])
    laterIdx = backtrackStates.index(invalidNodes[-1])
    
    formerPath = backtrackStates[0:formerIdx]
    laterPath = backtrackStates[laterIdx+1:]
    for node in invalidNodes:
        del childParentRelation[node]
        del c2cInfo[node]
    del childParentRelation[laterPath[0]]
    return formerPath, laterPath, childParentRelation, c2cInfo

# function to repair graph 
# if returns false, replan locally around the new obstacle
def attemptReconnection(sNew, gNew, newObsPosition):
    
    x_diff = gNew[0] - sNew[0]
    y_diff = gNew[1] - sNew[1]

    sub_points = []
    sub_points.append(sNew)

    if(np.abs(x_diff) > np.abs(y_diff)):
        diff = np.abs(x_diff)
    else:
        diff = np.abs(y_diff)

    for index in range(1, int(np.abs(diff))):
        point = (sNew[0] + (index * x_diff / np.abs(diff)), sNew[1] + (index * y_diff / np.abs(diff)))
        sub_points.append(point)

    for point in sub_points:
        if IsObstacle(point[0], point[1], gap = 10, dynamicPos = newObsPosition):
            return False
    return True

# function to run Informed RRT* and plan locally around the new obstacle
def planLocally(invalidNodes,newObsPosition,explored_states,backtrackStates,childParentRelation, c2cInfo):
    formerPath, laterPath, cpRelationUpdt, c2cUpdt = disconnectInvalidNodes(invalidNodes,backtrackStates,
                                                                            childParentRelation,c2cInfo)
    sLocal, gLocal = formerPath[-1], laterPath[0]
    reconnectFlag = attemptReconnection(sLocal, gLocal, newObsPosition)
    if reconnectFlag:
        print("found reconnection")
        cpRelationUpdt[gLocal] = sLocal
        c2cUpdt[gLocal] = c2cUpdt[sLocal] + np.sqrt(((gLocal[0] - sLocal[0]) ** 2) + ((gLocal[1] - sLocal[1]) ** 2))
        newBacktrackPath = formerPath + laterPath
        return (sLocal, gLocal, explored_states, newBacktrackPath,cpRelationUpdt, c2cUpdt), ()
    else:
        print("reconnection failed.. planning locally")
        rrtLocal = InformedRRTStar(sLocal, gLocal, newObsPosition)
        (localExploration, localBackTrack, localCPRelation, localc2c) = rrtLocal.Search_algorithm(20000)
        for i in invalidNodes:
            explored_states.remove(i)
        allStatesExplored = explored_states + localExploration
        allBackTrack = formerPath + localBackTrack[1:-1] + laterPath
        cpRelationUpdt.update(localCPRelation)
        c2cUpdt[gLocal] = c2cUpdt[sLocal] + localc2c[gLocal]
        return (sLocal, gLocal,localExploration, 
                localBackTrack, localCPRelation, localc2c), (allStatesExplored, allBackTrack, 
                                                             cpRelationUpdt, c2cUpdt)
    
    

# class definition for Informed RRT*

class InformedRRTStar(object):
    # init
    def __init__(self, start, goal, dynamicPos):
    
        self.start = start
        
        self.goal = goal
        
        self.xLength = 300
        self.yLength = 200
        
        self.clearance = 10
        
        self.c2c = {}
        
        self.path = {}
        
        self.goalThreshold = 10
        
        self.vertices = []
        
        self.stepSize = 9
        
        self.stepFactor = 10
        
        self.dynamicPos = dynamicPos
        
    # function to get eucledian distance
    def euclidean_dist(self, p1, p2):

        return (np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2)))
    
    
    # function to get random point generator
    def gen_rand_point(self, curr_Cbest, cMin, xCenter, C_matrix):

        if(curr_Cbest < float('inf')):
            
            L_matrix = np.diag([(curr_Cbest / 2.0), (np.sqrt((curr_Cbest ** 2) - (cMin ** 2)) / 2.0), (np.sqrt((curr_Cbest ** 2) - (cMin ** 2)) / 2.0)])
        
            a = random.random()
            b = random.random()
            
            if(b < a):
                a, b = b, a
            
            sample_point = (b * np.cos(2 * np.pi * a / b), b * np.sin(2 * np.pi * a / b))
            xBall = np.array([[sample_point[0]], [sample_point[1]], [0]])
            rand_point = np.dot(np.dot(C_matrix, L_matrix), xBall) + xCenter
            randX = round(rand_point[(0, 0)], 1)
            randY = round(rand_point[(1, 0)], 1)
        else:
            randX = round(random.uniform((self.clearance), (self.xLength  - self.clearance)), 1)
            randY = round(random.uniform((self.clearance), (self.yLength  - self.clearance)), 1)
        return (randX, randY)
    
    
    # function to nearest neighbour in the graph
    def get_nearest_neighbour(self, currX, currY):

        min_dist = float('inf')
        nearestVertex = -1
        
        for vertex in self.vertices:
            distance = self.euclidean_dist(vertex, (currX, currY))
            if(distance < min_dist):
                min_dist = distance
                nearestVertex = vertex
        
        return nearestVertex
    
    
    # function to check obstacle between points
    def Obstacle_between_vertices(self, point1, point2):
        
        x_diff = point2[0] - point1[0]
        y_diff = point2[1] - point1[1]
        
        sub_points = []
        sub_points.append(point1)
        
        if(np.abs(x_diff) > np.abs(y_diff)):
            diff = np.abs(x_diff)
        else:
            diff = np.abs(y_diff)
        
        for index in range(1, int(np.abs(diff))):
            point = (point1[0] + (index * x_diff / np.abs(diff)), point1[1] + (index * y_diff / np.abs(diff)))
            sub_points.append(point)
        
        for point in sub_points:
            if(IsObstacle(point[0], point[1], self.clearance, self.dynamicPos) or 
               IsValid(point[0], point[1], self.clearance) == False):
                return True
        return False
    
    
    # function to generate new node
    def get_new_node(self, x_rand, x_nearest):

        slope = (x_rand[1] - x_nearest[1]) / (x_rand[0] - x_nearest[0])
        factor = self.stepSize * np.sqrt(1.0 / (1.0 + (slope ** 2)))
        
        point_1 = (round(x_nearest[0] + factor, 1), round(x_nearest[1] + (slope * factor), 1))
        point_2 = (round(x_nearest[0] - factor, 1), round(x_nearest[1] - (slope * factor), 1))
        flag1 = False
        flag2 = False
        
        if(self.Obstacle_between_vertices(x_nearest, point_1)):
            flag1 = True
        if(self.Obstacle_between_vertices(x_nearest, point_2)):
            flag2 = True
        
        distance_1 = self.euclidean_dist(x_rand, point_1)
        distance_2 = self.euclidean_dist(x_rand, point_2)
        if(distance_1 < distance_2):
            return (flag1, point_1)
        else:
            return (flag2, point_2)
    
    
    # function to get nodes in the local neighbourhood
    def get_neighbourhood_vetices(self, x_new):

        neighbourhood_vertices = []
        for index in range(0, len(self.vertices)):
            dist = self.euclidean_dist(x_new, self.vertices[index])
            if(dist < self.stepFactor):
                neighbourhood_vertices.append(self.vertices[index])
        return neighbourhood_vertices
    
    
    # function to get the parent of new node in neighbourhood
    def get_neighbourhood_vetices_parent(self, neighbourhood, x_new):
        
        dist = self.c2c[neighbourhood[0]]
        parent = neighbourhood[0]
        for index in range(1, len(neighbourhood)):
            curr_dist = self.c2c[neighbourhood[index]] + self.euclidean_dist(neighbourhood[index], x_new)
            if(curr_dist < dist):
                dist = curr_dist
                parent = neighbourhood[index]
        return parent
    
    
    # function to implement RRT* algorithm 
    def Search_algorithm(self, iters):

        self.c2c[self.start] = 0
        self.vertices.append(self.start)
        pathTaken = []
        cBest = float('inf')
        pathLen = float('inf')
        cMin = np.sqrt((self.start[0] - self.goal[0]) ** 2 + (self.start[1] - self.goal[1]) ** 2)
        xCenter = np.matrix([[(self.start[0] + self.goal[0]) / 2.0], [(self.start[1] + self.goal[1]) / 2.0], [0]])
        M_matrix = np.dot(np.matrix([[(self.goal[0] - self.start[0]) / cMin], [(self.goal[1] - self.start[1]) / cMin], [0]]) , np.matrix([1.0, 0.0, 0.0]))
        U, S, V_T = np.linalg.svd(M_matrix, 1, 1)
        C_matrix = np.dot(np.dot(U, np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(V_T))])), V_T)

        for step in range(0, iters):
            print("\rProcessing.. %d%% complete"%(step / (iters) * 100), end="", flush=True)

            (x_rand_x, x_rand_y) = self.gen_rand_point(cBest, cMin, xCenter, C_matrix)
            x_rand = (x_rand_x, x_rand_y)

            (x_nearest_x, x_nearest_y) = self.get_nearest_neighbour(x_rand_x, x_rand_y)
            x_nearest = (x_nearest_x, x_nearest_y)
            
            if((x_nearest[0] == x_rand[0]) or (x_nearest[1] == x_rand[1])):
                continue
    
            (flag, x_new) = self.get_new_node(x_rand, x_nearest)
            if(flag == True):
                continue
            
            neighbourhood = self.get_neighbourhood_vetices(x_new)
            
            parent = self.get_neighbourhood_vetices_parent(neighbourhood, x_new)
            x_nearest = parent
            
            if(self.Obstacle_between_vertices(x_nearest, x_new)):
                continue
            
            self.vertices.append(x_new)
            self.path[x_new] = x_nearest
            self.c2c[x_new] = self.c2c[x_nearest] + self.euclidean_dist(x_nearest, x_new)
            
            for index in range(0, len(neighbourhood)):
                distance_from_start = self.c2c[x_new] + self.euclidean_dist(x_new, neighbourhood[index])
                if(distance_from_start < self.c2c[neighbourhood[index]]):
                    self.c2c[neighbourhood[index]] = distance_from_start
                    self.path[neighbourhood[index]] = x_new
            
            dist_from_goal = self.euclidean_dist(x_new, self.goal)
            if(dist_from_goal <= self.goalThreshold):
                self.path[self.goal] = x_new
                self.c2c[self.goal] = self.c2c[x_new] + self.euclidean_dist(self.goal, x_new)
                backtrack_vertices = self.goal
                
                temp_path = []
                temp_len = self.c2c[backtrack_vertices]
                while(backtrack_vertices != self.start):
                    temp_path.append(backtrack_vertices)
                    backtrack_vertices = self.path[backtrack_vertices]
                temp_path.append(self.start)
                temp_path = list(reversed(temp_path))
                
                if(cBest > temp_len):
                    cBest = temp_len
                    pathTaken = temp_path

                if round((cBest - cMin),2)< 20:
                    break
        # return explored, backtrack states, hierarchy dictionary, and 
        return (self.vertices, pathTaken, self.path, self.c2c)



# main code

import time

start_time= time.time()
# get start X-position
startX = float(input("Enter the x-coordinate for start node : "))

# get start Y-position
startY = float(input("Enter the y-coordinate for start node : "))

# get goal X-position
goalX = float(input("Enter the x-coordinate for goal node : "))

# get goal Y-position
goalY = float(input("Enter the y-coordinate for goal node : "))

# max iterations
iters= 50000
# take start and goal node as input
start = (startX,startY)
goal = (goalX,goalY)

# run infomed RRT* algorithn
rrt = InformedRRTStar(start, goal,[-1,-1])
# check for valid start and goal position
if(IsValid(start[0], start[1])):
    if(IsValid(goal[0], goal[1])):
        if(IsObstacle(start[0],start[1]) == False):
            if(IsObstacle(goal[0], goal[1]) == False):
                (explored_states, backtrack_states,childParentRelation, c2cInfo) = rrt.Search_algorithm(iters)
                
                # animate the path exploration
                animate(explored_states, backtrack_states, start, goal,childParentRelation)
                
                # generate new obstatacle randomly in planned path
                newObsPosition = generateRandomObstacle(backtrack_states)
                
                # validate path for new 
                invalidPathFlag, invalidNodes = validatePath(backtrack_states,newObsPosition)
                
                # check if path is invalid
                if invalidPathFlag:
                    # plan locally around the new obstacle
                    (sLocal, gLocal,localExp, localBt,localCP, localc2c),globalData = planLocally(invalidNodes,newObsPosition,explored_states, 
                                                                                 backtrack_states,childParentRelation, c2cInfo)
                    # animate the new exploration and path around the obstacle
                    animate(localExp, localBt, sLocal, gLocal ,localCP, dynamicPos = newObsPosition)
                    if len(globalData)>0:
                        # animate the new path from start to goal
                        animate(globalData[0], globalData[1], start, goal ,globalData[2], dynamicPos = newObsPosition)

            else:
                print("The entered goal node is an obstacle ")
        else:
            print("The entered start node is an obstacle ")
    else:
        print("The entered goal node outside the map ")
else:
    print("The entered start node is outside the map ")    
end_time=time.time()
print('Time taken to find an optimal path: ',end_time-start_time)





