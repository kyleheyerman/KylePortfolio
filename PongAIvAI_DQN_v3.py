#   PongAIvAI
#   Authors: Michael Guerzhoy and Denis Begun, 2014-2016.
#   http://www.cs.toronto.edu/~guerzhoy/
#   Email: guerzhoy at cs.toronto.edu
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version. You must credit the authors
#   for the original parts of this code.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   Parts of the code are based on T. S. Hayden Dennison's PongClone (2011)
#   http://www.pygame.org/project-PongClone-1740-3032.html

#   This code runs with Python 2 and requires PyGame for Python 2
#   Download PyGame here: https://bitbucket.org/pygame/pygame/downloads

# https://stackoverflow.com/questions/64209238/error-15-initializing-libiomp5md-dll-but-found-libiomp5md-dll-already-initial
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pygame, sys, time, random, os
from pygame.locals import *

import math
import argparse

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Copy your DQN class here.
"""
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        pass

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        pass

"""
Load your model here. 
"""
policy_net = torch.load('./model.pth', map_location=torch.device('cpu'))


"""
Replace the state in move method in the Paddle class by your state if you used a different one in your network.
"""

"""
Change other parameters in init_game function at the bottom of this file if necessary. 
"""


def select_action(state):
    with torch.no_grad():
        action = policy_net(state).max(1)[1].view(1, 1).item()
    if action == 0:
        action = "up"
    elif action == 1:
        action = "down"
    return action


stacked_states = None

ball_history = [(0,0)]

def stack_state(stacked_states, state):
    if stacked_states is None: # if it is None
        stacked_states = np.stack(arrays=[state, state, state, state])
        stacked_states = stacked_states
    else:
        stacked_states[0] = stacked_states[1]
        stacked_states[1] = stacked_states[2]
        stacked_states[2] = stacked_states[3]
        stacked_states[3] = state
    
    return stacked_states

def stack_states(states, state):
    states = stack_state(states, state)
    return states

class fRect:
    """Like PyGame's Rect class, but with floating point coordinates"""

    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1 #self.size > 0 and other_frect.size > 0


class Paddle:
    def __init__(self, pos, size, speed, max_angle,  facing, timeout):
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.facing = facing
        self.max_angle = max_angle
        self.timeout = timeout

    def factor_accelerate(self, factor):
        self.speed = factor*self.speed

    def move(self, enemy_frect, ball_frect, table_size, enemy): # add an enemy
        
        # The program crashes if move_getter crashes. The runtime of 
        # move_getter is not limited
        # direction = self.move_getter(self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size))
        
        # The program continues if move_getter crashes. The runtime of
        # move_getter is limited

        # Without this line, it has this error: https://stackoverflow.com/questions/10851906/python-3-unboundlocalerror-local-variable-referenced-before-assignment
        global stacked_states

        if enemy:
            direction = timeout(self.move_getter, (self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size)), {}, self.timeout)
        else: # DQN agent
            """
            Replace the state below by your state if you used a different one in your network.
            """
            state = (self.frect.pos[0], self.frect.pos[1], self.frect.size[0], self.frect.size[1],
                     enemy_frect.pos[0], enemy_frect.pos[1], enemy_frect.size[0], enemy_frect.size[1],
                     ball_frect.pos[0], ball_frect.pos[1], ball_frect.size[0], ball_frect.size[1],
                     table_size[0], table_size[1])
            
            stacked_states = stack_states(stacked_states, state)
            state = torch.tensor(stacked_states, dtype=torch.float32, device=device).unsqueeze(0)
            direction = select_action(state)

        if direction == "up":
            self.frect.move_ip(0, -self.speed)
        elif direction == "down":
            self.frect.move_ip(0, self.speed)

        to_bottom = (self.frect.pos[1]+self.frect.size[1])-table_size[1]

        if to_bottom > 0:
            self.frect.move_ip(0, -to_bottom)
        to_top = self.frect.pos[1]
        if to_top < 0:
            self.frect.move_ip(0, -to_top)


    def get_face_pts(self):
        return ((self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1]),
                (self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1] + self.frect.size[1]-1)
                )

    def get_angle(self, y):
        center = self.frect.pos[1]+self.size[1]/2
        rel_dist_from_c = ((y-center)/self.size[1])
        rel_dist_from_c = min(0.5, rel_dist_from_c)
        rel_dist_from_c = max(-0.5, rel_dist_from_c)
        sign = 1-2*self.facing

        return sign*rel_dist_from_c*self.max_angle*math.pi/180

class Ball:
    def __init__(self, table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag):
        rand_ang = (.4+.4*random.random())*math.pi*(1-2*(random.random()>.5))+.5*math.pi # half chance positive or negative 
        speed = (init_speed_mag*math.cos(rand_ang), init_speed_mag*math.sin(rand_ang)) # random init speed at two directions 
        pos = (table_size[0]/2, table_size[1]/2)
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.prev_bounce = None

    def get_center(self):
        return (self.frect.pos[0] + .5*self.frect.size[0], self.frect.pos[1] + .5*self.frect.size[1])

    def get_speed_mag(self):
        return math.sqrt(self.speed[0]**2+self.speed[1]**2)

    def factor_accelerate(self, factor):
        self.speed = (factor*self.speed[0], factor*self.speed[1])

    def move(self, paddles, table_size, move_factor):
        moved = 0
        walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]

        for wall_rect in walls_Rects:
            if self.frect.get_rect().colliderect(wall_rect):
                c = 0
                
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                r1 = 1+2*(random.random()-.5)*self.dust_error
                r2 = 1+2*(random.random()-.5)*self.dust_error

                self.speed = (self.wall_bounce*self.speed[0]*r1, -self.wall_bounce*self.speed[1]*r2)
                
                while c > 0 or self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1              

        for paddle in paddles:
            if self.frect.intersect(paddle.frect):
                if (paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[0]/2) or \
                (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0]/2):
                    continue
                
                c = 0
                
                while self.frect.intersect(paddle.frect) and not self.frect.get_rect().colliderect(walls_Rects[0]) and not self.frect.get_rect().colliderect(walls_Rects[1]):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    
                    c += 1
                theta = paddle.get_angle(self.frect.pos[1]+.5*self.frect.size[1])
                
                v = self.speed

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]

                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to the  increase by 1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    self.speed = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    self.speed = (v[0], v[1])
                self.prev_bounce = paddle               

                while c > 0 or self.frect.intersect(paddle.frect):
                
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    
                    c -= 1
                
                moved = 1
                
        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)

def directions_from_input(paddle_rect, other_paddle_rect, ball_rect, table_size):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        return "up"
    elif keys[pygame.K_DOWN]:
        return "down"
    else:
        return None

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        print("TIMEOUT")
        return default
    else:
        return it.result

def render(screen, paddles, ball, score, table_size):
    screen.fill(black)

    pygame.draw.rect(screen, white, paddles[0].frect.get_rect())
    pygame.draw.rect(screen, white, paddles[1].frect.get_rect())

    pygame.draw.circle(screen, white, (int(ball.get_center()[0]), int(ball.get_center()[1])),  int(ball.frect.size[0]/2), 0)

    pygame.draw.line(screen, white, [screen.get_width()/2, 0], [screen.get_width()/2, screen.get_height()])

    score_font = pygame.font.Font(None, 32)
    screen.blit(score_font.render(str(score[0]), True, white), [int(0.4*table_size[0])-8, 0])
    screen.blit(score_font.render(str(score[1]), True, white), [int(0.6*table_size[0])-8, 0])

    pygame.display.flip()

def check_point(score, ball, table_size):
    if ball.frect.pos[0]+ball.size[0]/2 < 0:
        score[1] += 1
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        return (ball, score)
    elif ball.frect.pos[0]+ball.size[0]/2 >= table_size[0]:
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        score[0] += 1
        return (ball, score)

    return (ball, score)
    
def game_loop(screen, paddles, ball, table_size, clock_rate, turn_wait_rate, score_to_win, display):
    score = [0, 0]

    while max(score) < score_to_win:
        old_score = score[:]
        ball, score = check_point(score, ball, table_size)
        paddles[0].move(paddles[1].frect, ball.frect, table_size, False) # not enemy
        paddles[1].move(paddles[0].frect, ball.frect, table_size, True) # enemy
        
        inv_move_factor = int((ball.speed[0]**2+ball.speed[1]**2)**.5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                ball.move(paddles, table_size, 1./inv_move_factor)
        else:
            ball.move(paddles, table_size, 1)      
        
        if not display:
            continue
        if score != old_score:
            font = pygame.font.Font(None, 32)
            if score[0] != old_score[0]:
                screen.blit(font.render("Left scores!", True, white, black), [0, 32])
            else:
                screen.blit(font.render("Right scores!", True, white, black), [int(table_size[0]/2+20), 32])

            pygame.display.flip()
            clock.tick(turn_wait_rate)

        render(screen, paddles, ball, score, table_size)

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_q]:
            return

        clock.tick(clock_rate)

    if display:
        font = pygame.font.Font(None, 64)
        if score[0] > score[1]:
            screen.blit(font.render("Left wins!", True, white, black), [24, 32])
        else:
            screen.blit(font.render("Right wins!", True, white, black), [24, 32])
        pygame.display.flip()
        clock.tick(2)

    pygame.event.pump()
    while any(pygame.key.get_pressed()):
        pygame.event.pump()
        clock.tick(30)

    print(score)

def init_game():
    table_size = (440, 280)
    paddle_size = (10, 70)
    ball_size = (15, 15)
    paddle_speed = 1
    max_angle = 45

    paddle_bounce = 1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 2
    timeout = 0.0003
    clock_rate = 80
    turn_wait_rate = 3
    """
    Change 10 to another number like 100 if testing the agents using more rounds of game in one episode. 
    """
    score_to_win = 10

    """
    Change the display to choose showing the game or only the final scores. 
    If playing for many rounds like 100 rounds, set the display as False. 
    """
    display = True

    screen = pygame.display.set_mode(table_size)
    pygame.display.set_caption('PongAIvAI')

    paddles = [Paddle((20, table_size[1]/2), paddle_size, paddle_speed, max_angle,  1, timeout),
               Paddle((table_size[0]-20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 0, timeout)]
    ball = Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)
    
    """
    Change the enemy AI to another AI below if testing against another AI. 
    """
    right_ai = __import__("chaser_ai")
    paddles[1].move_getter = right_ai.pong_ai
                        
    game_loop(screen, paddles, ball, table_size, clock_rate, turn_wait_rate, score_to_win, display)
    
    pygame.quit()


if __name__ == '__main__':
    pygame.init()

    init_game()