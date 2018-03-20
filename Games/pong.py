from game import Game

class Pong(Game):
    def __init__(self, duration):
        super(Pong, self).__init__(duration)
    
    def start(self):
        return

    def draw(self):
        return

    def execute(self, action):
        return
import pygame
from pygame.locals import *
import sys, os, traceback
import random
from math import *
if sys.platform == 'win32' or sys.platform == 'win64':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.display.init()
pygame.font.init()
pygame.mixer.init(buffer=0)

screen_size = [800,600]
icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
pygame.display.set_caption("Ping Pang Pong - v.3.0.0 - Ian Mallett - 2012")
surface = pygame.display.set_mode(screen_size)

font = {
    18 : pygame.font.SysFont("Times New Roman",18),
    72 : pygame.font.SysFont("Times New Roman",72)
}

def rndint(x):
    return int(round(x))
def clamp(x, minimum,maximum):
    if x < minimum: return minimum
    if x > maximum: return maximum
    return x

class Paddle:
    def __init__(self, x,y,w,h, key_l,key_r,key_d,key_u):
        self.pos = [x,y]
        self.dim = [w,h]
        
        self.key_l = key_l
        self.key_r = key_r
        self.key_d = key_d
        self.key_u = key_u
    def move(self, rel_x,rel_y):
        self.pos[0] += dt*rel_x
        self.pos[1] -= dt*rel_y
        self.pos[0] = clamp(self.pos[0],0,screen_size[0]-self.dim[0])
        self.pos[1] = clamp(self.pos[1],0,screen_size[1]-self.dim[1])
    def update(self,key):
        speed = 300
        if self.key_l!=None and key[self.key_l]: self.move(-speed, 0)
        if self.key_r!=None and key[self.key_r]: self.move( speed, 0)
        if self.key_d!=None and key[self.key_d]: self.move( 0,-speed)
        if self.key_u!=None and key[self.key_u]: self.move( 0, speed)
    def draw(self,color):
        pygame.draw.rect(surface,        color,(self.pos[0],self.pos[1],self.dim[0],self.dim[1]),0)
        pygame.draw.rect(surface,(255,255,255),(self.pos[0],self.pos[1],self.dim[0],self.dim[1]),1)
class Player:
    def __init__(self,color,paddles):
        self.score = 0
        self.color = color
        self.paddles = list(paddles)
    def add_score(self):
        self.score += 1
players = [
    Player((0,255,0), [Paddle(               5,   screen_size[1]/2-30,10,60, None,None,   K_s, K_w)]),
    Player((0,0,255), [Paddle(screen_size[0]-5-10,screen_size[1]/2-30,10,60, None,None,K_DOWN,K_UP)])
]
        
class Ball:
    def __init__(self, x,y, speed):
        self.pos = [x,y]
        self.trail = []

        angle = pi / 2
        while abs(cos(angle)) < 0.1 or abs(cos(angle)) > 0.9:
            angle = radians(random.randint(0,360))
        self.speed = [speed*cos(angle),speed*sin(angle)]
        
        self.radius = 2
    def update(self):
        self.trail = [self.pos + [self.radius]] + self.trail
        while len(self.trail) > 10: self.trail = self.trail[:-1]
    def move(self, dt):
        self.pos[0] += dt*self.speed[0]
        self.pos[1] += dt*self.speed[1]
    def speed_up(self):
        factor = 1.1
        self.speed[0] *= factor
        self.speed[1] *= factor
    def draw(self):
        light = 255/10
        for px,py,r in self.trail[::-1]:
            pygame.draw.circle(surface,(light,0,0),list(map(rndint,[px,py])),r)
            light += 255/10
        pygame.draw.circle(surface,(255,255,255),list(map(rndint,self.pos)),self.radius)
balls = []

def get_input():
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if   event.type == QUIT: return False
        elif event.type == KEYDOWN:
            if   event.key == K_ESCAPE: return False
    for player in players:
        for paddle in player.paddles:
            paddle.update(keys)
    return True
def update():
    for ball in balls:
        ball.update()
between_rounds_timer = 3.0
def move():
    global balls, between_rounds_timer

    balls2 = []
    for ball in balls:
        removed = False
        for substep in range(10): #Do substeps so that it is much harder for the ball to ghost through the paddles.
            ball.move(dt/10.0)
            
            if ball.pos[0] < 0:
                players[1].add_score()
                removed = True
                break
            elif ball.pos[0] > screen_size[0]:
                players[0].add_score()
                removed = True
                break

            if ball.pos[1] < 0:
                ball.pos[1] = 0
                ball.speed[1] *= -1
            elif ball.pos[1] > screen_size[1]:
                ball.pos[1] = screen_size[1]
                ball.speed[1] *= -1
            for player in players:
                for paddle in player.paddles:
                    if ball.pos[0] > paddle.pos[0] and ball.pos[0] < paddle.pos[0]+paddle.dim[0] and\
                       ball.pos[1] > paddle.pos[1] and ball.pos[1] < paddle.pos[1]+paddle.dim[1]:
                        dist_lrdu = [
                            ball.pos[0] - paddle.pos[0],
                            (paddle.pos[0]+paddle.dim[0]) - ball.pos[0],
                            (paddle.pos[1]+paddle.dim[1]) - ball.pos[1],
                            ball.pos[1] - paddle.pos[1],
                        ]
                        dist_min = min(dist_lrdu)
                        if   dist_min == dist_lrdu[0]: ball.speed[0] = -abs(ball.speed[0])
                        elif dist_min == dist_lrdu[1]: ball.speed[0] =  abs(ball.speed[0])
                        elif dist_min == dist_lrdu[2]: ball.speed[1] =  abs(ball.speed[1])
                        elif dist_min == dist_lrdu[3]: ball.speed[1] = -abs(ball.speed[1])
                        ball.speed_up()
                        
        if not removed: balls2.append(ball)

    if len(balls2) == 0 and len(balls) > 0: #someone scored the last of the balls
        between_rounds_timer = 3.0

    balls = balls2
    if len(balls) == 0:
        between_rounds_timer -= dt
        if between_rounds_timer < 0:
            balls.append(Ball(screen_size[0]/2,screen_size[1]/2,200.0))
def draw():
    surface.fill((0,0,0))

    for ball in balls:
        ball.draw()
    for player in players:
        for paddle in player.paddles:
            paddle.draw(player.color)
            
    p1_score_text = font[18].render("Score "+str(players[0].score),True,(255,255,255))
    p2_score_text = font[18].render("Score "+str(players[1].score),True,(255,255,255))
    surface.blit(p1_score_text,(                                         20,20))
    surface.blit(p2_score_text,(screen_size[0]-p2_score_text.get_width()-20,20))

    if between_rounds_timer > 0:
        alpha = between_rounds_timer - int(between_rounds_timer)
        alpha = rndint(255*alpha)

        count = font[72].render(str(int(between_rounds_timer)+1),True,(alpha,alpha,alpha))
        
        sc = 0.5  *  (1.0 + between_rounds_timer-int(between_rounds_timer))
        count = pygame.transform.smoothscale(count,list(map(rndint,[count.get_width()*sc,count.get_height()*sc])))
        
        surface.blit(count,(screen_size[0]/2-count.get_width()/2,screen_size[1]/2-count.get_height()/2))
    
    pygame.display.flip()

def main():
    global dt
    dt = 1.0/60.0
    
    clock = pygame.time.Clock()
##    pygame.mouse.set_visible(False)
    while True:
        if not get_input(): break
        update()
        move()
        draw()
        clock.tick(60)
        dt = 1.0/clamp(clock.get_fps(),30,90)
##    pygame.mouse.set_visible(True)
    pygame.quit(); sys.exit()
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        tb = sys.exc_info()[2]
        traceback.print_exception(e.__class__, e, tb)
        pygame.quit()
        input()
        sys.exit()
