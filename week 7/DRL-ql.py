# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:13:00 2019
@author: Zhibo QU
Q learning agent with fake goal and observer
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
OBS = [[3,3],[3,5]]
for i in range(0, X):
    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
BARRIERS = OBS + WALLS
#print(len(BARRIERS))
#GOALS = [[5,5]]
START = [3,0]
FAKE = [[0,7]] #fake goal
GOALS = [[8,5]]


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
        time.sleep(0.02)
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
        
    def seeObs(self,state):
        suc = self.getSuccessor(state)
        for next in suc:
            if next in OBS:
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
    while state not in agent.goals:
        action = agent.choose_action(str(state))
        while env.hitBarrier (state,action):
            action = agent.choose_action(str(state))
        notLearning = False
        if env.seeObs(state):
            if agent.observation:
                notLearning = True
        new_state, r = env.env_reaction(agent, state, action, False)
        if (agent.findDistance(new_state,GOALS[0])+agent.findDistance(new_state,FAKE[0]) == agent.findDistance(GOALS[0],FAKE[0])):
            notLearning = True
        if not notLearning:
            r += observer.myReward(str(state),action)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state

def training(agent, observer):
    t = 0
    while (t < 10*EPISODES):
        show(agent,1)
        run_agent (agent, observer)
        run_agent (observer, agent)
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
    
env.mainloop()