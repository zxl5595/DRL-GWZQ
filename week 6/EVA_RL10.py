# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 03:16:16 2019

@author: Gaoxiang
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
        self.file_exist = False
        
        if observation:
            self.goals = FAKE
            if self.path_ob.exists():
                self.q_table = pd.read_pickle('observer.pickle')
                self.file_exist = True
            else:
                self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)                
        else:
            self.goals = GOALS
            if self.path_rl.exists():
                self.q_table = pd.read_pickle('agent.pickle')
                self.file_exist = True
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

class eva(object):#1: score for early stopping 2: decide: deceptive path
    def __init__(self, start = START, fake_goals = FAKE, real_goals = GOALS, barriers = BARRIERS, x_total = X, y_total = Y, path = []):
        self.start = start
        self.fake_goals = fake_goals
        self.real_goals = real_goals
        self.barriers = barriers
        self.x = x_total
        self.y = y_total
        self.path = path
        
    def cal_score(self):
        cost = len(self.path)
        visited_LDP = False
        LDPs = []
        """
        for goal in self.real_goals:
            opt = self.distance(self.start, goal)
            if goal in self.path:
                e_finished = 0.5
            for fake in self.fake_goals:
                LDPs += self.getLDP(self.start, goal, fake)
                """
        goal = self.real_goals[0]
        fake = self.fake_goals[0]
        LDPs = self.getLDP(self.start, goal, fake)
        opt = self.distance(self.start, goal)
        e_LDP = 0
        e_cost = 0
        e_FTS = 0
        e_finished = 0
        opt_ldp = 1000
        first_truth = False
        for ldp in LDPs:
            if opt_ldp > self.distance(self.start, ldp):
                opt_ldp = self.distance(self.start, ldp)
            if ldp in self.path:
                visited_LDP = True
                e_LDP = 0.5
        if cost == (opt+1) and visited_LDP:
            e_cost = 1
        if self.path[-1] in self.real_goals:
            e_finished = 0.5
        for i in range(len(self.path)-1):
            state = self.path[i]
            new_state = self.path[i + 1]
            if self.distance(state, goal)>self.distance(new_state, goal) and self.distance(state, fake)<self.distance(new_state, fake) and not first_truth :
                e_FTS = round((self.distance(self.start, state)/ opt_ldp )*4, 2)
                first_truth = True
        
        score = e_LDP + e_cost + e_FTS + e_finished
   
        return score
    
    def distance (self, position1, position2):
        dist = abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])
        return dist

    def getLDP(self, start, goal, fake):
        beta = (self.distance(goal, fake) + self.distance(start, goal) - self.distance(start, fake)) / 2
        LDP = []
        for a in range(9):
            for b in range(9):
                ldp = [a, b]
                if (self.distance(ldp, goal) == beta) and (self.distance(ldp, fake) == self.distance(goal, fake) - self.distance(ldp, goal)):
                    LDP.append(ldp)
        return LDP


    def decision(self):
        if self.cal_score() == 6:
            return True
        else:
            return False


def ini_agent(agent):
    if agent.file_exist == False:
        state = START
        while state not in agent.goals:
            action = agent.choose_action(str(state))
            new_state, r = env.env_reaction(agent, state, action, False)
            agent.learn(str(state), action, r, str(new_state))
            state = new_state
    else:
        pass

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
    text = tk.Text(env, height = 10, width = 30)
    text.pack(side = 'top')
    times = 0
    text.insert(tk.INSERT, "Trained Times :"+str(times)+'\n' )
    text.insert(tk.INSERT, "Last Evaluation :" + str(0))
    for j in range(EPISODES):
        run_agent (agent, observer)
        run_agent (observer, agent)
        temp_path = get_path(agent)
        times += 1
        ENV = eva(path = temp_path)
        text.delete(1.0,tk.END)
        text.insert(tk.INSERT, "Trained Times :"+str(times)+'\n' )
        text.insert(tk.INSERT, "Last Evaluation :" + str(ENV.cal_score()))
        if ENV.cal_score() >= 6:
            print('early stop')
            break
        else:
            continue
       
        
    print (agent.q_table)
    print(ENV.cal_score())
    #agent.q_table.to_pickle('q_table.pickle')
    #agent.q_table.to_csv('q_table.csv')


def show(agent,e):
    agent.epsilon = 2
    path = []
    for t in range (e):
        env.reset()
        env.render()
        state = START
        if t == 0:
            path.append(state)
        else:
            pass
        while state not in agent.goals:
            action = agent.choose_action(str(state))
            new_state, r = env.env_reaction(agent, state, action, True)
            state = new_state
            if t == 0:
                path.append(state)
            else:
                pass
    agent.epsilon = EPSILON
    env.reset()
    return path

def get_path(agent):
    agent.epsilon = 2
    path = []

    env.reset()
    env.render()
    state = START
    path.append(state)

    while state not in agent.goals:
        action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(agent, state, action, True)
        state = new_state
        path.append(state)

    agent.epsilon = EPSILON
    env.reset()
    return path


if __name__ == "__main__":
    env = Maze(SIZE, X, Y)
    RL = Qlearning_Agent(False)
    OB = Qlearning_Agent(True)
    RL.epsilon = 0
    OB.epsilon = 0
    path = []
    iniTraining(RL)
    print ("RL trained")
    print (RL.q_table)
    #RL.q_table.to_pickle('agent.pickle')
    #RL.q_table.to_csv('agent.csv')
    iniTraining(OB)
    print ("OB trained")
    #OB.q_table.to_pickle('observer.pickle')
    #OB.q_table.to_csv('observer.csv')
    #print (OB.q_table)
    #env.after(100, update)
    RL.epsilon = EPSILON
    OB.epsilon = EPSILON
    training(RL,OB)
    final_path = show(RL,5)
    print(final_path)
    EVA = eva(path = final_path)

    if EVA.decision():
        print('this is a good deceptive path')
    else:
        print('this is not a good deceptive path\n')    
    print("the score of this path is " + str(EVA.cal_score()))
    env.mainloop()
