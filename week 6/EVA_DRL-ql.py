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

#actions
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
ID_ACTIONS = list(range(len(ACTIONS)))# 0:left, 1:right, 3:up, 4:down
#q_learning
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 2
LAMBDA = 0.9
EPISODES = 30
#map
SIZE = 70
X = 9
Y = 9
WALLS = [[-X, -1], [-1, Y], [X, Y], [X, -1]]
for i in range(0, X):
    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
BARRIERS = [] + WALLS
#print(len(BARRIERS))
#GOALS = [[5,5]]
START = [3,0]
FAKE = [[0,8]] #fake goal
GOALS = [[8,6]]
        
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

    state = START
    while state not in agent.goals:
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
        new_state, r = env.env_reaction(agent, state, action, False)
        r += observer.myReward(str(state),action)
        #print (r)
        agent.learn(str(state), action, r, str(new_state))
        state = new_state

def training(agent, observer):
    text = tk.Text(env, height = 10, width = 30)
    text.pack(side = 'left')
    times = 1
    e_LDP = 0
    e_cost = 0
    e_finished = 0
    e_FTS = 0
    message = " "
    text.insert(tk.INSERT, "Training Times : 0\n" )
    text.insert(tk.INSERT, "Total Evaluation : "+str(e_LDP + e_cost + e_FTS+e_finished))
    while True:
        (e_LDP, e_cost, e_finished, e_FTS) = show(agent, 1)
        text.delete(1.0,tk.END)
        run_agent (agent, observer)
        run_agent (observer, agent)
        text.insert(tk.INSERT, "Training Times :"+str(times)+'\n' )
        """
        text.insert(tk.INSERT, "Find LDP : "+str(e_LDP)+'\n')
        text.insert(tk.INSERT, "First Truth Step : "+str(e_FTS)+'\n')
        text.insert(tk.INSERT, "Deceptive Path Cost : "+str(e_cost)+'\n')
        text.insert(tk.INSERT, "Finished : "+str(e_finished)+'\n')
        """
        text.insert(tk.INSERT, "Total Evaluation : "+str(e_LDP + e_cost + e_FTS+e_finished))
        times += 1

    print (agent.q_table)

def distance (position1, position2):
    dist = abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])
    return dist

def getLDP(start, goal, fake):
    beta = (distance(goal, fake) + distance(start, goal) - distance(start, fake)) / 2
    LDP = []
    for a in range(9):
        for b in range(9):
            ldp = [a, b]
            if (distance(ldp, goal) == beta) and (distance(ldp, fake) == distance(goal, fake) - distance(ldp, goal)):
                LDP.append(ldp)
    return LDP


def show(agent,e):
    agent.epsilon = 2
    text = tk.Text(env, height = 4, width = 30)
    text.pack()
    for t in range (e):
        text.delete('1.0','end')
        env.reset()
        env.render()
        state = START
        goal = [8,6]
        fake = [0,8]
        LDPs = getLDP(START, goal, fake)
        cost = 0
        c_to_LDP = 100000
        e_LDP = 0
        e_cost = 0
        e_FTS = 0
        for LDP in LDPs:
            if c_to_LDP > distance(state, LDP):
                c_to_LDP = distance(state, LDP)
        visited_LDP = False
        first_truth = False
        while state not in agent.goals:
            cost += 1
            action = agent.choose_action(str(state))
            new_state, r = env.env_reaction(agent, state, action, True)
            if new_state in LDPs:
                e_LDP = 0.5
                visited_LDP = True
            #text.insert('insert', "Find LDP : "+str(e_LDP)+'\n')
            print(e_LDP)
            if distance(state, goal)>distance(new_state, goal) and distance(state, fake)<distance(new_state, fake) and not first_truth:
                e_FTS = round(((cost-1) / c_to_LDP) *4, 2)
                first_truth = True
            #text.insert('insert', "First Truth Step : "+str(e_FTS)+'\n')
            print(e_FTS)
            if new_state in agent.goals:
                e_finished = 0.5
            state = new_state
            if cost == distance(START, goal) - 1 and visited_LDP:
                e_cost = 1
            #text.insert('insert', "Deceptive Path Cost : "+str(e_cost)+'\n')
            print(e_cost)
            #text.insert('insert', "Total Evaluation : "+str(e_LDP + e_cost + e_FTS))
        #time.sleep(0.3)
            #text.delete('1.0','end')
            """
        text.insert('insert', "Find LDP : "+str(e_LDP)+'\n')
        text.insert('insert', "First Truth Step : "+str(e_FTS)+'\n')
        text.insert('insert', "Deceptive Path Cost : "+str(e_FTS)+'\n')
        text.insert('insert', "Total Evaluation : "+str(e_LDP + e_cost + e_FTS))
        time.sleep(0.2)
        """
    agent.epsilon = EPSILON
    env.reset()
    print (e_LDP, e_cost, e_finished, e_FTS)
    return (e_LDP, e_cost, e_finished, e_FTS)
    #text.delete('1.0','end')





if __name__ == "__main__":
    env = Maze(SIZE, X, Y)
    RL = Qlearning_Agent(False)
    OB = Qlearning_Agent(True)
    RL.epsilon = 0
    OB.epsilon = 0
    print ("Initializing...")
    iniTraining(RL)
    #print ("RL trained")
    print (RL.q_table)
    iniTraining(OB)
    #print ("OB trained")
    #print (OB.q_table)
    #env.after(100, update)
    RL.epsilon = EPSILON
    OB.epsilon = EPSILON
    training(RL,OB)
    show(RL,1)
    
env.mainloop()