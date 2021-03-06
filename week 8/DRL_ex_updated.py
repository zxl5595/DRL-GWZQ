
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:13:00 2019
@author: Zhibo QU
has been updated on Apirl 28th
Q learning agent with fake goal and observer, using simulation strategy.
"""

import numpy as np
import pandas as pd
import time
import copy
import tkinter as tk

#actions
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
ID_ACTIONS = list(range(len(ACTIONS)))# 0:left, 1:right, 2:up, 3:down
#q_learning
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.9
LAMBDA = 0.9
EPISODES = 30
#map
SIZE = 100
X = 9
Y = 9
WALLS = [[-X, -1], [-1, Y], [X, Y], [X, -1]]
OBS = []
for i in range(0, X):
    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
BARRIERS = OBS + WALLS
#print(len(BARRIERS))
#GOALS = [[5,5]]

def distance (position1, position2):
    dist = abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])
    return dist


while True:
    random_x = np.random.randint(0, X-1, 3)
    random_y = np.random.randint(0, Y-1, 3)
    random_s = [random_x[0], random_y[0]]
    random_f = [random_x[1], random_y[1]]
    random_r = [random_x[2], random_y[2]]
    random_list = [random_s, random_f, random_r]
    """
    for i in range(len(random_list)):
        if random_list.count(random_list[i]) == 1:
            pass
        else:
            continue
    """
    if (distance(random_list[0],random_list[1])>=(X+Y)/3) and (distance(random_list[1],random_list[2])>=(X+Y)/3) and (distance(random_list[0],random_list[2])>=(X+Y)/3):
        for i in range(len(random_list)):
            if random_list.count(random_list[i]) == 1:
                pass
            else:
                continue
    else:
        continue
    break
    
START = random_s
FAKE = [random_f] #fake goal
GOALS = [random_r]
final_actions = []
final_path = [START]

class Queue:
    def __init__(self):
        self.list = []

    def push(self,item):
        self.list.insert(0,item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0
        
class Maze(tk.Tk, object):
    def __init__(self, size, x, y):
        super(Maze, self).__init__()
        self.title('maze')              
        self.goals = GOALS
        self.fake = FAKE
        self.obstacle = OBS
        self.barriers = BARRIERS
        self.size = size
        self.x_total = x
        self.y_total = y
        self.geometry('1800x900')
        self.start = START
        self._build_maze()
        
        
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg = 'white', height = self.size * self.y_total, 
                                width = self.size * self.x_total)
        
        for i in range(self.x_total):
            for j in range(5):
                self.canvas.create_line(self.size - 2 + j + self.size*i, 0, 
                                        self.size - 2 + j + self.size*i, self.size*self.y_total)
                
        for i in range(self.y_total):
            for j in range(5):
                self.canvas.create_line(0, self.size - 2 + j + self.size*i,
                                        self.size*self.y_total, self.size - 2 + j + self.size*i)

#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
#        print(11111)
#        food_file = tk.PhotoImage(file='food.gif')
#        self.food = self.canvas.create_image(1510, 10, anchor='nw', image = food_file)
        self.food = self.canvas.create_rectangle(10 + self.goals[0][0]*self.size, 10 + self.goals[0][1]*self.size,(self.goals[0][0] + 1)*self.size - 10, (self.goals[0][1] + 1)*self.size - 10,fill = 'red')
        self.fakefood = self.canvas.create_rectangle(10 + self.fake[0][0]*self.size, 10 + self.fake[0][1]*self.size,(self.fake[0][0] + 1)*self.size - 10, (self.fake[0][1] + 1)*self.size - 10,fill = 'orange')
        self.startpos = self.canvas.create_rectangle(10 + self.start[0]*self.size, 10 + self.start[1]*self.size,(self.start[0] + 1)*self.size - 10, (self.start[1] + 1)*self.size - 10,fill = 'green')
        
        self.mouse = self.canvas.create_oval(10 + self.start[0]*self.size, 10 + self.start[1]*self.size, 10 + self.start[0]*self.size + self.size - 20, 10 +self.start[1]*self.size + self.size - 20, fill = 'black')
        if (len(self.obstacle)>0):
            self.ob = []
            for i in range(len(self.obstacle)):
                self.ob.append(self.canvas.create_rectangle(10 + self.obstacle[i][0]*self.size, 10 + self.obstacle[i][1]*self.size,(self.obstacle[i][0] + 1)*self.size - 10, (self.obstacle[i][1] + 1)*self.size - 10,fill = 'gray'))
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.mouse)
        self.mouse = self.canvas.create_oval(10 + self.start[0]*self.size, 10 + self.start[1]*self.size, 10 + self.start[0]*self.size + self.size - 20, 10 +self.start[1]*self.size + self.size - 20, fill = 'black')
#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
        # return observation
        #return self.canvas.coords(self.rect)
        
    def move_to(self, action):
        self.update()
        time.sleep(0.1)
        # 0:left, 1:right, 2:up, 3:down
        if action == ID_ACTIONS[0]:
            self.canvas.move(self.mouse, -self.size, 0)
        elif action == ID_ACTIONS[1]:
            self.canvas.move(self.mouse, self.size, 0)
        elif action == ID_ACTIONS[2]:
            self.canvas.move(self.mouse, 0, -self.size)
        else:
            self.canvas.move(self.mouse, 0, self.size)
            
    def render(self):
        time.sleep(0.01)
        self.update()
    
    def getSuccessor(self, state):
        new_state = copy.copy(state)
        left = [new_state[0] - 1,new_state[1]]
        right = [new_state[0] + 1,new_state[1]]
        up = [new_state[0],new_state[1] - 1]
        down = [new_state[0],new_state[1] + 1]
        return [left,right,up,down]

    def hitBarrier(self,state,action):
        suc = self.getSuccessor(state)
        new_state = suc[action]
        if new_state in BARRIERS:
            return True
        return False

    def env_reaction(self, agent, state, action, show):
        suc = self.getSuccessor(state)
        new_state = suc[action]
        if new_state not in agent.goals:
            r = 0
        else:
            r = 1            
        if show:
            self.move_to(action)
        return new_state, r



class Qlearning_Agent:
    def __init__(self, observation, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA, e_greedy = EPSILON):
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.observation = observation
        if observation:
            self.goals = FAKE
        else:
            self.goals = GOALS
        
    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            scores_of_actions = self.q_table.loc[state, :]
            action = np.random.choice(scores_of_actions[scores_of_actions 
                                                        == np.max(scores_of_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def myReward (self, state, action):
        self.check_state_exist(state)
        scores_of_actions = self.q_table.loc[state, :]
        myActions = scores_of_actions[scores_of_actions == np.max(scores_of_actions)].index
        #print(action, myActions)
        r = -0.2
        if (action in myActions):
            r = 0.2
        return r

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )
                
    def learn(self, state, action, r, new_state):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if new_state not in self.goals:
            q_target = r + self.gamma * self.q_table.loc[new_state, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def findDistance(self,startState, finalState):
        positions = Queue()
        paths = Queue()
        pathToPosition = []
        visited = []
        positions.push(startState)
        currentState = positions.pop()
        while not (currentState == finalState):
            if currentState not in visited:
                visited.append(currentState)
                suc = env.getSuccessor(currentState)
                for st in suc:
                    if ((st not in visited) and (st not in BARRIERS)):
                        path = pathToPosition + [st]
                        positions.push(st)
                        paths.push(path)
            currentState = positions.pop()
            pathToPosition = paths.pop()
        return len(pathToPosition)
        

def ini_agent(agent):

    state = START
    while state not in agent.goals:
        action = agent.choose_action(str(state))
        while env.hitBarrier (state,action):
            action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, False)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state


def iniTraining(agent):
    for t in range(3*EPISODES):
        ini_agent(agent)


def run_agent(agent, observer):
    state = START
    while state not in observer.goals:
        action = observer.choose_action(str(state))
        while env.hitBarrier (state,action):
            action = observer.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, False)
        r += observer.myReward(str(state),action)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state
    while state not in agent.goals:
        action = agent.choose_action(str(state))
        while env.hitBarrier (state,action):
            action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, False)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state

def training(agent, observer):
    t = 0
    while (t < 1000):
        #show (agent,1)
        run_agent (agent, observer)
        #run_agent (observer, agent)
        t += 1
    print(agent.q_table)


def show(agent,e):
    agent.epsilon = 2
    for t in range (e):
        env.reset()
        env.render()
        state = START
        visited = []
        duplicated = []
        while state not in agent.goals:
            action = agent.choose_action(str(state))

            new_state, r = env.env_reaction(agent, state, action, True)
            if t == 0:
                final_actions.append(ACTIONS[action])
                final_path.append(new_state)
            state = new_state
            if state not in visited:
                 visited.append(state)
            elif state not in duplicated:
                duplicated.append(state)
            else:
                env.reset()
                break
            
    agent.epsilon = EPSILON
    env.reset()



#cost difference for state
def cosdiff_st(state,start,goal,fake):
    cosdiff_goal = distance(state,goal) - distance(start,goal)
    cosdiff_fake = distance(state,fake) - distance(start,fake)
    return (cosdiff_goal,cosdiff_fake)

#cost difference for action
def cosdiff_act(state,next_state,start,goal,fake):
    cosdiff_goal_st = distance(state,goal) - distance(start,goal)
    cosdiff_goal_nst = distance(next_state,goal) - distance(start,goal)
    cosdiff_fake_st = distance(state,fake) - distance(start,fake)
    cosdiff_fake_nst = distance(next_state,fake) - distance(start,fake)
    return (cosdiff_goal_st,cosdiff_goal_nst,cosdiff_fake_st,cosdiff_fake_nst)

def experiment(START, FAKE, GOALS, final_path):
    actions = []
    for i in range(len(final_path) - 1):
        actions.append([final_path[i], final_path[i + 1]])
    
    count_actions = len(actions)
    count_dissim_act = 0 #dissimulation action
    count_sim_act = 0 #simulation action
    count_true_act = 0 #truthful action
    count_state = len(final_path)
    count_dissim_st = 0 #dissimulation state
    count_sim_st = 0 #simulation state
    count_true_st = 0 #truthful state
    first_truthful_state = list()
    first_truthful_act = list()
    first_find_st = False
    first_find_act = False
    
    for i in range(len(actions)):
        states =actions[i]
        cosdiff_goal_st,cosdiff_goal_nst,cosdiff_fake_st,cosdiff_fake_nst = cosdiff_act(states[0],states[1],START,GOALS[0],FAKE[0])
        if (cosdiff_goal_st <= cosdiff_goal_nst) and (cosdiff_fake_st > cosdiff_fake_nst):
            count_sim_act += 1
        elif ((cosdiff_goal_st < cosdiff_goal_nst) and (cosdiff_fake_st < cosdiff_fake_nst)) or ((cosdiff_goal_st > cosdiff_goal_nst) and (cosdiff_fake_st > cosdiff_fake_nst)) or ((cosdiff_goal_st == cosdiff_goal_nst) and (cosdiff_fake_st == cosdiff_fake_nst)):
            count_dissim_act += 1
        else:
            count_true_act += 1
            if not first_find_act:
                first_truthful_act = states
                first_find_act = True
    
    for i in range(len(final_path)):
        cosdiff_goal,cosdiff_fake = cosdiff_st(final_path[i],START,GOALS[0],FAKE[0])
        if cosdiff_goal > cosdiff_fake:
            count_sim_st += 1
        elif not cosdiff_goal < cosdiff_fake:
            count_dissim_st += 1
        else:
            count_true_st += 1
            if not first_find_st:
                first_truthful_state = final_path[i]
                first_find_st = True

    
    rate_dissim_act = count_dissim_act/count_actions
    rate_sim_act = count_sim_act/count_actions
    rate_true_act = count_true_act/count_actions
    rate_dissim_st = count_dissim_st/count_state
    rate_sim_st = count_sim_st/count_state
    rate_true_st = count_true_st/count_state

    print ("action: ",rate_dissim_act, rate_sim_act,rate_true_act)
    print ("state: ",rate_dissim_st,rate_sim_st,rate_true_st)
    print ("first truthful act: ",first_truthful_act)
    print ("first truthful state: ",first_truthful_state)
    
    return None
    
    
if __name__ == "__main__":
    env = Maze(SIZE, X, Y)
    RL = Qlearning_Agent(False)
    OB = Qlearning_Agent(True)
    RL.epsilon = 0
    OB.epsilon = 0
    print ("Initializing...")
    iniTraining(RL)
    print (RL.q_table)
    iniTraining(OB)
    print (OB.q_table)
    RL.epsilon = EPSILON
    OB.epsilon = EPSILON
    training(RL,OB)
    show(RL,1)
    experiment(START, FAKE, GOALS, final_path)
    print ("final path: ",final_path)
    env.mainloop()
    
    
