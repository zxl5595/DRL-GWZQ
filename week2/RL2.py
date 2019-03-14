# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:21:56 2019

@author: Gaoxiang
"""

import numpy as np
import pandas as pd
import time
import tkinter as tk

ACTIONS = ['LEFT', 'RIGHT']# 0:left, 1:right
TOTAL_SPACE = 6
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.9

        
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('maze')
        self.geometry('1800x300')
        self._build_maze()
        self.goal = [5]

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height = 300, width = 1800)
        
        for i in range(5):
            for j in range(5):
                self.canvas.create_line(298+ j + 300*i, 0, 298 + j + 300*i, 300)

#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
#        print(11111)
#        food_file = tk.PhotoImage(file='food.gif')
#        self.food = self.canvas.create_image(1510, 10, anchor='nw', image = food_file)
        self.food = self.canvas.create_rectangle(1510, 10, 1510+280, 10+280, fill='red')
        self.mouse = self.canvas.create_oval(10, 10, 10 + 280, 10 + 280, fill='black')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.mouse)
        self.mouse = self.canvas.create_oval(10, 10, 10 + 280, 10 + 280, fill='black')
#        mouse_file = tk.PhotoImage(file='mouse.gif')
#        self.mouse = self.canvas.create_image(10, 10, anchor='nw', image = mouse_file)
        # return observation
        #return self.canvas.coords(self.rect)
        
    def move_to(self, action):
        self.update()
        time.sleep(0.5)
        if action == 1:
            self.canvas.move(self.mouse, 300, 0)
        else:
            self.canvas.move(self.mouse, -300, 0)

    def render(self):
        time.sleep(0.1)
        self.update()
        
    def env_reaction(self, state, action):
        new_state = state
        if action == 1:
            if state <= 4:
                new_state = state + 1
                self.move_to(action)
            else:
                pass
        else:
            if state >= 1:
                new_state = state - 1
                self.move_to(action)
            else:
                pass
        if new_state == 5:
            r = 1
        else:
            r = 0
        return new_state, r

class Qlearning_Agent:
    def __init__(self, actions, learning_rate = 0.1, reward_decay = GAMMA, e_greedy = EPSILON):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def choose_action(self, state):
        self.check_state_exist(state)
        if np.random.uniform() < self.epsilon:
            scores_of_actions = self.q_table.loc[state, :]
            action = np.random.choice(scores_of_actions[scores_of_actions == np.max(scores_of_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
                
    def learn(self, state, action, r, new_state):
        self.check_state_exist(new_state)
        q_predict = self.q_table.loc[state, action]
        if new_state not in env.goal:
            q_target = r + self.gamma * self.q_table.loc[new_state, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


def run_agent():
    env.reset()
    state = 0
    RL = Qlearning_Agent(actions = list(range(len(ACTIONS))))
    while state != 5:
        action = RL.choose_action(state)
        new_state, r = env.env_reaction(state, action)
        RL.learn(state, action, r, new_state)
        state = new_state
            
    #return q_table
def update():
    for t in range(10):
        env.reset()
        #while state != 5:
        env.render()
        #steps = rl()
        
#        print('\r\nQ-table:\n')
#        print(q_table)
#        print('Episode %s: total_steps = %s' % (t+1, steps))
        run_agent()
#        print('game over')
#        env.destroy()
    env.reset()
            
if __name__ == "__main__":
    env = Maze()
    update()
    #env.after(100, update)
    env.mainloop()