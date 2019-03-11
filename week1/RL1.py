# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:22:13 2019

@author: Gaoxiang
"""

import numpy as np
import pandas as pd
import time
import tkinter as tk

ACTIONS = ['LEFT', 'RIGHT']
TOTAL_SPACE = 6
GAMMA = 0.9
ALPHA = 0.1
q_table = pd.DataFrame(np.zeros((TOTAL_SPACE, len(ACTIONS))), columns = ACTIONS)

        
class (tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.title('maze')
        self.geometry('1800x300')
        self._build_maze()

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
        if action == 'RIGHT':
            self.canvas.move(self.mouse, 300, 0)
        else:
            self.canvas.move(self.mouse, -300, 0)

    def render(self):
        time.sleep(0.1)
        self.update()




def choose_action(state, q_table):
    score_of_actions = q_table.iloc[state, :]
    if np.random.uniform() > 0.9 or (score_of_actions == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        action = score_of_actions.idxmax()
    return action

def state_update(state, action):
    new_state = state
    if action == 'RIGHT':
        if state <= 4:
            new_state = state + 1
            env.move_to(action)
        else:
            pass
    else:
        if state >= 1:
            new_state = state - 1
            env.move_to(action)
        else:
            pass
    if new_state == 5:
        r = 1
    else:
        r = 0
    return new_state, r

def rl():
    #for episode in range(10):
        env.reset()
        state = 0
        steps = 0
        while state != 5:
            action = choose_action(state, q_table)
            new_state, r = state_update(state, action)
            q_predict = q_table.loc[state, action]
            if new_state != 5:
                q_target = r + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = r
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = new_state
            steps += 1
        return steps
            #print(state)
            #print('\n')
#        print('\r\nQ-table:\n')
#        print(q_table)
        
            
    #return q_table
def update():
    for t in range(10):
        env.reset()
        #while state != 5:
        env.render()
        steps = rl()
        print('\r\nQ-table:\n')
        print(q_table)
        print('Episode %s: total_steps = %s' % (t+1, steps))
#        print('game over')
#        env.destroy()
    env.reset()
            
if __name__ == "__main__":
    env = Maze()
    update()
    #env.after(100, update)
    env.mainloop()