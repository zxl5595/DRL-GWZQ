# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:13:00 2019
Updated on Mon Apr 1  15:33:58 2019
@author: Xianglin ZHOU
Q learning agent with fake goal and observer
"""

import numpy as np
import pandas as pd
import time
import copy
import tkinter as tk
import math

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
#SIZE = 100
SIZE = 70
X = 9
Y = 9
WALLS = [[-X, -1], [-1, Y], [X, Y], [X, -1]]
for i in range(0, X):
    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
BARRIERS = [] + WALLS
#print(len(BARRIERS))
#GOALS = [[5,5]]
FAKE = [[6,4]] #fake goal
#FAKE = [[7,2]]
GOALS = [[4,6]]

STEP = 3
        
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
        self.geometry('1800x1800')
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
        self.mouse = self.canvas.create_oval(10, 10, 10 + self.size - 20, 10 + self.size - 20, fill = 'black')
#        self.barriers = self.canvas.create_rectangle(10 + 3*self.size, 10 + 3*self.size,
#                                                 (3 + 1)*self.size - 10, (3 + 1)*self.size - 10, 
#                                                 fill = 'blue')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.mouse)
        self.mouse = self.canvas.create_oval(10, 10, 
                                             10 + self.size - 20, 10 + self.size - 20, 
                                             fill = 'black')
#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
        # return observation
        #return self.canvas.coords(self.rect)
        
    def move_to(self, action):
        self.update()
        time.sleep(0.05)
        #time.sleep(0.5)
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
        
class Observer(object):    
    def __init__(self, init_state = [], current_state = [], goals = [], barriers = []):
        self.init_state = init_state
        self.current_state = current_state
        self.goals = goals
        self.barriers = barriers
    
    """
    def env_reaction(self, state, action):
        new_state = copy.copy(state)
        if action == ID_ACTIONS[0]:
            new_state[0] -= 1
        elif action == ID_ACTIONS[1]:
            new_state[0] += 1
        elif action == ID_ACTIONS[2]:
            new_state[1] -= 1
        else:
            new_state[1] += 1
            
        if new_state in self.barriers:
            new_state = state
        else:
            env.move_to(action)
        if new_state in self.goals:
            r = 1
        else:
            r = 0
        return new_state, r
    """
    
    #give reward based on the difference of distance to the fake goal and real goal
    def distance (self, position1, position2):
        dist = abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])
        return dist

    """
    def env_reaction(self, state, action):
        new_state = copy.copy(state)
        ##new_state = state
        ## 0:left, 1:right, 2:up, 3:down
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
        else:
            env.move_to(action)
        if not RL.simulation:
            if new_state in FAKE:
                r = 0.1
                RL.simulation = True
            elif (self.distance(state, FAKE[0]) >= self.distance(new_state, FAKE[0])):
                r = 0.01
            else:
                r = -0.01
        else:
            if new_state in GOALS:
                r = 1
            elif (self.distance(state, GOALS[0]) >= self.distance(new_state, GOALS[0])):
                r = 0.1
            else:
                r = -0.1
        
        return new_state, r
    """
    
    
    def env_reaction(self, state, action, move):
        new_state = copy.copy(state)
        #new_state = state
        # 0:left, 1:right, 2:up, 3:down
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
        else:
            if move:
                env.move_to(action)
        
        """
        if not RL.simulation:
            if new_state in FAKE:
                r = 0.1
                RL.simulation = True
            elif (self.distance(state, FAKE[0]) >= self.distance(new_state, FAKE[0])):
                r = 0.02 + 0.01 * (self.distance(new_state, FAKE[0])- self.distance(new_state, GOALS[0]))
            else:
                r = -0.01
        else:
            if new_state in GOALS:
                r = 1
            elif (self.distance(state, GOALS[0]) >= self.distance(new_state, GOALS[0])):
                r = 0.2
            else:
                r = -0.1
        
        """
        if new_state in GOALS:
            r = 1
        else:
            r = 0.01 * (2*(self.distance(state, GOALS[0]) - self.distance(new_state, GOALS[0])) + (self.distance(new_state, GOALS[0]) - self.distance(new_state, FAKE[0])))
        
        return new_state, r


    
class RL_Agent(object):
    def __init__(self, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA, e_greedy = EPSILON):
        self.actions = actions 
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
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
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )
    def learn(self, *args):
        pass

class Qlearning_Agent(RL_Agent):
    def __init__(self, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA, e_greedy = EPSILON):
        super(Qlearning_Agent, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.simulation = False
                
    def learn(self, state, action, r, new_state):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if new_state not in GOALS:
            q_target = r + self.gamma * self.q_table.loc[new_state, :].max()
        else:
            q_target = r
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


class Sarsa_Agent(RL_Agent):
    def __init__(self, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA, e_greedy = EPSILON):
        super(Sarsa_Agent, self).__init__(actions, learning_rate, reward_decay, e_greedy)
                
    def learn(self, state, action, r, new_state, new_action):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if new_state not in GOALS:
            q_target = r + self.gamma * self.q_table.loc[new_state, new_action]
        else:
            q_target = r
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

class SARSA_TD_Agent (RL_Agent): #n step sarsa
    def __init__(self, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA,
                 e_greedy = EPSILON, step = STEP):
        super(SARSA_TD_Agent, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.step = step
    
    def calculate_Q (self, state, action,move):
        new_state, r = Obs.env_reaction(state, action, move)
        self.check_state_exist(str(new_state))
        new_action = RL.choose_action(str(new_state))
        q_predict = self.q_table.loc[str(state), action]
        if new_state not in GOALS:
           q_target = r + self.gamma * self.q_table.loc[str(new_state), new_action]
        else:
           q_target = r
        self.q_table.loc[str(state), action] += self.lr * (q_target - q_predict)

    
    def learn(self, state, action):
        state_tau = state
        action_tau = action
        T = 10000000000
        R = []
        for t in range(self.step):
            move = False
            if t < T:
                next_state, r = Obs.env_reaction(state, action, move)
                self.check_state_exist(str(next_state))
                #print (t, self.q_table)
                R.append(r)
                if next_state in GOALS:
                    T = t + 1
                else:
                    action = RL.choose_action(str(next_state))
                    state = next_state
                if t == self.step - 1:
                   step_state = state
                   step_action = action
                   #print (self.q_table.loc[str(step_state), step_action])
                   self.calculate_Q(step_state,step_action,move)
    
    
        tau = t - self.step + 1
        if tau >= 0:
            G = 0
            for i in range(tau+1,min(tau+self.step,T)):
                G = G + math.pow(self.gamma,i-tau-1) * R[i-tau-1]
                if tau + self.step < T:
                    G = G + math.pow(self.gamma,self.step) * self.q_table.loc[str(step_state), step_action]
                    self.q_table.loc[str(state_tau), action_tau] = self.q_table.loc[str(state_tau), action_tau] + self.lr * (G - self.q_table.loc[str(state_tau), action_tau])
            



class N_Step_Sarsa_Agent(RL_Agent): #sarsa lambda
    def __init__(self, actions = ID_ACTIONS, learning_rate = ALPHA, reward_decay = GAMMA,
                 e_greedy = EPSILON, trace_decay = LAMBDA):
        super(N_Step_Sarsa_Agent, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        self.lambd = trace_decay
        self.eligibility_trace = self.q_table.copy()
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
            
    def learn(self, state, action, r, new_state, new_action):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if new_state not in GOALS:
            q_target = r + self.gamma * self.q_table.loc[new_state, new_action]
        else:
            q_target = r
            
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1
        self.q_table += self.lr * (q_target - q_predict) * self.eligibility_trace #!!!the whole table
        self.eligibility_trace *= self.gamma * self.lambd

def run_agent():
    env.reset()
    state = [0, 0]
    RL.simulation = False # Q learning
    
    while state not in GOALS:       
        action = RL.choose_action(str(state))
        
        RL.learn(state, action) #N STEP SARSA
        move = True
        new_state, r = Obs.env_reaction(state, action, move)
        new_action = RL.choose_action(str(new_state))
        state = new_state
    print(RL.q_table)
"""

def run_agent():
    env.reset()
    state = [0, 0]
    RL.simulation = False # Q learning
    action = RL.choose_action(str(state))
    RL.learn(state, action) #N STEP SARSA
"""

"""
def run_agent():
    env.reset()
    state = [0, 0]
    ##action = RL.choose_action(str(state)) #Sarsa
    RL.simulation = False # Q learning
    
    while state not in GOALS:
        action = RL.choose_action(str(state)) #Qlearning
        new_state, r = Obs.env_reaction(state, action)
        ##new_action = RL.choose_action(str(new_state))
        RL.learn(str(state), action, r, str(new_state)) #Qlearning
        ##RL.learn(str(state), action, r, str(new_state), new_action) #Sarsa
        state = new_state
    ##action = new_action #Sarsa
    print(RL.q_table)
"""

def training():
    for t in range(EPISODES):
        env.reset()
        env.render()

        run_agent()
#        print('game over')
#        env.destroy()
    env.reset()
            
if __name__ == "__main__":
    env = Maze(SIZE, X, Y)
    Obs = Observer(goals = env.goals, barriers = env.barriers)
    #RL = Qlearning_Agent()
    #RL = Sarsa_Agent()
    #RL = N_Step_Sarsa_Agent()
    RL = SARSA_TD_Agent() #n step SARSA
    training()
    #env.after(100, update)
    env.mainloop()
