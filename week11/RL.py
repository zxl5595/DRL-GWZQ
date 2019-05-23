# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:39:11 2019

@author: Gaoxiang
"""

import numpy as np
import pandas as pd
import time
import copy
import tkinter as tk
import json

ARGUMENT_FILE = 'argument.json'

#actions
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
ID_ACTIONS = list(range(len(ACTIONS)))# 0:left, 1:right, 3:up, 4:down
#q_learning
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.9
EPISODES = 50
#map
#SIZE = 40
#X = 15
#Y = 15
#WALLS = [[-X, -1], [-1, Y], [X, Y], [X, -1]]
#OBS = [[4,5], [11,11], [5,5], [5,4], [11,12], [12,11], [13,11], [14, 11], [11,13]]
#for i in range(0, X):
#    WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
#BARRIERS = WALLS
#START = [6,0]
#FAKE = [[0,13]] #fake goal
#GOALS = [[11,14]]
#
#final_actions = []
#final_path = [START]
        
class Maze(tk.Tk, object):
    def __init__(self, size, x, y,wall,barriers,GOALS,FAKE,START,OBS):
        super(Maze, self).__init__()
        self.title('maze')              
        self.goals = GOALS
        self.fake = FAKE
        self.start = START
        self.obstacle = OBS
        self.size = size
        self.x_total = x
        self.y_total = y
        self.wall = wall
        self.barriers = barriers
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
        self.mouse = self.canvas.create_oval(10 + self.start[0]*self.size, 10 + self.start[1]*self.size, 10 + self.start[0]*self.size + self.size - 20, 10 +self.start[1]*self.size + self.size - 20, fill = 'black')
        if (len(self.obstacle)>0):
            self.ob = []
            for i in range(len(self.obstacle)):
                self.ob.append(self.canvas.create_rectangle(10 + self.obstacle[i][0]*self.size, 10 + self.obstacle[i][1]*self.size,(self.obstacle[i][0] + 1)*self.size - 10, (self.obstacle[i][1] + 1)*self.size - 10,fill = 'gray'))
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
        time.sleep(0.01)
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
        
    
    def env_reaction(self, state, action):
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
            
        if new_state in self.barriers:
            new_state = state
        else:
            self.move_to(action)
        if new_state in GOALS:
            r = 1
        else:
            r = 0
        return new_state, r


class Qlearning_Agent:
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


def run_agent(RL):
    env.reset()
    state = env.start
 
    while state not in GOALS:
#        print(state)
        action = RL.choose_action(str(state))
        new_state, r = env.env_reaction(state, action)
        RL.learn(str(state), action, r, str(new_state))
        state = new_state
    #print(RL.q_table)
            
    #return q_table
def training(RL,file_name):
    for t in range(EPISODES):
        env.reset()
        env.render()
        run_agent(RL)
#        print('game over')
#        env.destroy()
    env.reset()

def getPath(agent):
    agent.epsilon = 2
    state = env.start
    path = [state]
    while state not in GOALS:
        print(state)
        action = agent.choose_action(str(state))
        new_state, r = env.env_reaction(state, action)
        state = new_state
        path.append(state)
    return path
    

def distance (position1, position2):
    dist = abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])
    return dist

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

def experiment(START, FAKE, GOALS, final_path, file_name):
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
    last_deceptive_st = list()
    last_deceptive_act = list()
#    last_de_st = False
#    last_de_act = False
    st_list = []
    act_list = []
    
    for i in range(len(actions)):
        states =actions[i]
        cosdiff_goal_st,cosdiff_goal_nst,cosdiff_fake_st,cosdiff_fake_nst = cosdiff_act(states[0],states[1],START,GOALS[0],FAKE[0])
        if (cosdiff_goal_st <= cosdiff_goal_nst) and (cosdiff_fake_st > cosdiff_fake_nst):
            count_sim_act += 1
            act_list.append("Sim")
        elif ((cosdiff_goal_st < cosdiff_goal_nst) and (cosdiff_fake_st < cosdiff_fake_nst)) or ((cosdiff_goal_st > cosdiff_goal_nst) and (cosdiff_fake_st > cosdiff_fake_nst)) or ((cosdiff_goal_st == cosdiff_goal_nst) and (cosdiff_fake_st == cosdiff_fake_nst)):
            count_dissim_act += 1
            act_list.append("Dis")
        else:
            count_true_act += 1
            act_list.append("Tru")
            if not first_find_act:
                first_truthful_act = states
                first_find_act = True
    
    for i in range(len(final_path)):
        cosdiff_goal,cosdiff_fake = cosdiff_st(final_path[i],START,GOALS[0],FAKE[0])
        if cosdiff_goal > cosdiff_fake:
            count_sim_st += 1
            st_list.append("Sim")
        elif not cosdiff_goal < cosdiff_fake:
            count_dissim_st += 1
            st_list.append("Dis")
        else:
            count_true_st += 1
            st_list.append("Tru")
            if not first_find_st:
                first_truthful_state = final_path[i]
                first_find_st = True
    
    for i in range(len(st_list)):
        f = False
        for j in st_list[i:]:
            if j == "Tru":
                f = True
                continue
            else:
                f = False
                break
        if f:
            last_deceptive_st = final_path[i - 1]
            break
    if not f:
        last_deceptive_st = final_path[-1]

    for i in range(len(act_list)):
        f = False
        for j in act_list[i:]:
            if j == "Tru":
                f = True
                continue
            else:
                f = False
                break
        if f:
            last_deceptive_act.append(final_path[i-1])
            last_deceptive_act.append(final_path[i])
            break
    if not f:
        last_deceptive_act.append(final_path[-2])
        last_deceptive_act.append(final_path[-1])
        
    rate_dissim_act = round(count_dissim_act/count_actions, 3)
    rate_sim_act = round(count_sim_act/count_actions, 3)
    rate_true_act = round(count_true_act/count_actions, 3)
    rate_dissim_st = round(count_dissim_st/count_state, 3)
    rate_sim_st = round(count_sim_st/count_state, 3)
    rate_true_st = round(count_true_st/count_state, 3)

#    print ("Step amount: ", count_actions+1)
#    print ("Truthful state amount: ", count_true_st)
#    print ("Truthful action amount: ", count_true_act)
#    print ("Simulation state amount: ", count_sim_st)
#    print ("Simulation action amount: ", count_sim_act)
#    print ("Dissimulation state amount: ", count_dissim_st)
#    print ("Dissimulation action amount: ", count_dissim_act)
#    print ("Last Deceptive state: ", last_deceptive_st)
#    print ("Last Deceptive action: ", last_deceptive_act)
#    print ("action: ",rate_dissim_act, rate_sim_act,rate_true_act)
#    print ("state: ",rate_dissim_st,rate_sim_st,rate_true_st)
#    print ("first truthful act: ",first_truthful_act)
#    print ("first truthful state: ",first_truthful_state)
#    print ("action list: ", act_list)
#    print ("state list: ", st_list)
#    print ("(s, f, r): ", (START, FAKE, GOALS[0]))

    result_dict = {'Step amount':str(count_actions+1),'Truthful state amount':str(count_true_st),
                   'Truthful action amount':str(count_true_act),'Simulation state amount':str(count_sim_st),
                   'Simulation action amount':str(count_sim_act),'Dissimulation state amount':str(count_dissim_st),
                   'Dissimulation action amount':str(count_dissim_act),'Last Deceptive state':str(last_deceptive_st),
                   'Last Deceptive action':str(last_deceptive_act),'action':str((rate_dissim_act, rate_sim_act,rate_true_act)),
                   'state':str((rate_dissim_st,rate_sim_st,rate_true_st)),'first truthful act':str(first_truthful_act),
                   'first truthful state':str(first_truthful_state),'action list':str(act_list),'state list':str(st_list),
                   '(s, f, r)':str((START, FAKE, GOALS[0]))}
    result_json = json.dumps(result_dict)

    with open (file_name,'w') as f:
        f.write(result_json)
        
        return None

def main(exp,exp_count):
    count = 3
    for i in range(count):

        index = str(i+1)
        file_name = './result_tru/' + str(exp_count) + '_'  + index + '.json'
        RL = Qlearning_Agent()
        training(RL,file_name)
        final_path = getPath(RL)
        experiment(START, FAKE, GOALS, final_path,file_name)
        print ("final path: ",final_path)       
#        experiment(START, FAKE, GOALS, final_path,file_name)
#        print ("final path: ",final_path)
    env.destroy()
    env.mainloop()

if __name__ == "__main__":
    with open (ARGUMENT_FILE) as a:
        exp_count = 1
        arg = json.load(a)
        arguments = arg["arguments"]
        for exp in arguments:
#            RL = Qlearning_Agent()
#            id = exp["id"]
            xy = exp["size"]
            X = xy[0]
            Y = xy[1]
            SIZE = 40
            START = exp["start"]
            FAKE = [exp["fake"]] #fake goal
            goal = exp["real"]
#            goal_x = goal[0]
#            goal_y = goal[1]
            GOALS = [goal]
            OBS = exp["wall"]
            WALLS = [[-X, -1], [-1, Y], [X, Y], [X, -1]]
            for i in range(0, X):
                WALLS = WALLS + [[i , -1]] + [[-1, i]] + [[i, Y]] + [[X, i]]
            BARRIERS = []+WALLS
            env = Maze(SIZE, X, Y,WALLS,BARRIERS,GOALS,FAKE,START,OBS)
            main(exp,exp_count)
            exp_count = exp_count + 1


#    env = Maze(SIZE, X, Y)
#    RL = Qlearning_Agent()
#    training()
#    env.mainloop()