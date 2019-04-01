# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:13:00 2019
@author: Xianglin ZHOU
Q learning agent with fake goal and observer
"""

import numpy as np
import pandas as pd
import time
import copy
import tkinter as tk
import pathlib

#actions
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
ID_ACTIONS = list(range(len(ACTIONS)))# 0:left, 1:right, 3:up, 4:down
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
for i in range(0, X):
    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
BARRIERS = [] + WALLS
#print(len(BARRIERS))
#GOALS = [[5,5]]
START = [1,0]
FAKE = [[0,7]] #fake goal
GOALS = [[7,5]]
        
class Maze(tk.Tk, object):
    def __init__(self, size, x, y):
        super(Maze, self).__init__()
        self.title('maze')              
        self.goals = GOALS
        self.fake = FAKE
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
#       self.barriers = self.canvas.create_rectangle(10 + self.barriers[0][0]*self.size, 10 + self.barriers[0][1]*self.size,(self.barriers[0][0] + 1)*self.size - 10, (self.barriers[0][1] + 1)*self.size - 10,fill = 'gray')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.mouse)
        self.mouse = self.canvas.create_oval(10 + self.start[0]*self.size, 10 + self.start[1]*self.size, 10 + self.start[0]*self.size + self.size - 20, 10 +self.start[1]*self.size + self.size - 20, fill = 'black')
#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
        # return observation
        #return self.canvas.coords(self.rect)
        
    def move_to(self, action):
        self.update()
        time.sleep(0.1)
        if action == ID_ACTIONS[0]:
            self.canvas.move(self.mouse, -self.size, 0)
        elif action == ID_ACTIONS[1]:
            self.canvas.move(self.mouse, self.size, 0)
        elif action == ID_ACTIONS[2]:
            self.canvas.move(self.mouse, 0, -self.size)
        else:
            self.canvas.move(self.mouse, 0, self.size)
            
    def render(self):
        time.sleep(0.1)
        self.update()
    
    def env_reaction(self, agent, state, action, show):
        new_state = copy.copy(state)
        #new_state = state
        if action == ID_ACTIONS[0]:
            new_state[0] -= 1
        elif action == ID_ACTIONS[1]:
            new_state[0] += 1
        elif action == ID_ACTIONS[2]:
            new_state[1] -= 1
        else:
            new_state[1] += 1
            
        if new_state in BARRIERS:
            new_state = state
        elif show:
            self.move_to(action)
        if new_state not in agent.goals:
            r = 0
        else:
            r = 1
        return new_state, r


class Qlearning_Agent:
    def __init__(self, observation, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA, e_greedy = EPSILON):
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        #self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.observation = observation
        self.path_ob = pathlib.Path('observer.pickle')
        self.path_rl = pathlib.Path('agent.pickle')
        
        if observation:
            self.goals = FAKE
            if self.path_ob.exists():
                self.q_table = pd.read_pickle('observer.pickle')
            else:
                self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)                
        else:
            self.goals = GOALS
            if self.path_rl.exists():
                self.q_table = pd.read_pickle('agent.pickle')
            else:
                self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) 
                
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
        r = -0.1
        if (action in myActions):
            r = 0.1
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
        if new_state not in GOALS:
            q_target = r + self.gamma * self.q_table.loc[new_state, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


def ini_agent(agent):
#载入已有的Q表可以省略此步骤
    state = START
    while state not in agent.goals:
        action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, False)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state
#输出Q表以供之后的学习载入

def iniTraining(agent):
    for t in range(3*EPISODES):
        ini_agent(agent)


def run_agent(agent, observer):
    state = START
    while state not in agent.goals:
        action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, False)
        r += observer.myReward(str(state),action)
        #print (r)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state

def training(agent, observer):
    show(agent, 5)
    for j in range(EPISODES):
        run_agent (agent, observer)
        run_agent (observer, agent)
    print (agent.q_table)
    agent.q_table.to_pickle('q_table.pickle')
    agent.q_table.to_csv('q_table.csv')
    
def show(agent,e):
    agent.epsilon = 2
    for t in range (e):
        env.reset()
        env.render()
        state = START
        while state not in agent.goals:
            action = agent.choose_action(str(state))
            new_state, r = env.env_reaction(agent, state, action, True)
            state = new_state
    agent.epsilon = EPSILON
    env.reset()

if __name__ == "__main__":
    env = Maze(SIZE, X, Y)
    RL = Qlearning_Agent(False)
    OB = Qlearning_Agent(True)
    RL.epsilon = 0
    OB.epsilon = 0
    iniTraining(RL)
    print ("RL trained")
    print (RL.q_table)
    RL.q_table.to_pickle('agent.pickle')
    RL.q_table.to_csv('agent.csv')
    iniTraining(OB)
    print ("OB trained")
    OB.q_table.to_pickle('observer.pickle')
    OB.q_table.to_csv('observer.csv')
    #print (OB.q_table)
    #env.after(100, update)
    RL.epsilon = EPSILON
    OB.epsilon = EPSILON
    training(RL,OB)
    show(RL,5)
    
env.mainloop()