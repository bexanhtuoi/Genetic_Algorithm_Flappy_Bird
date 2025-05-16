import asyncio
import sys

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from .ANN import ANN
from src.ANN.Multilayer import ANN2
from .GA import GA
import numpy as np
import time

class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            # sounds=Sounds(),
        )
        self.font = pygame.font.SysFont(None, 24) 
        self.start_time = None

    async def start(self):
        self.Population = 1000
        self.ANN = ANN(self.Population, 6, 3, 1)
        self.GA = GA(self.Population, 0.5)
        self.max = 0
        for i in range(1000):
            self.gen = i
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.players = [
                Player(self.config, x=int(self.config.window.width * 0.2), y=int((self.config.window.height - self.config.images.player[0].get_height()) / 2))
                for _ in range(self.Population)
            ]
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            self.start_time = pygame.time.get_ticks()
            await self.play()
            self.ANN.weight = self.GA.fit(self.fitness, self.ANN.weight)
            max = np.max(self.fitness)
            if max > self.max:
                self.max = max
            

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    def draw_text(self, text, x, y):
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.config.screen.blit(text_surface, (x, y))

    async def play(self):
        self.score.reset()
        for player in self.players:
            player.set_mode(PlayerMode.NORMAL)

        self.fitness = np.zeros(self.Population)

        while True:
            self.location = np.zeros(self.Population)
            self.speed = np.zeros(self.Population)
            self.distance = np.zeros(self.Population)
            self.upper = np.zeros(self.Population)
            self.under = np.zeros(self.Population)
            self.gap = np.zeros(self.Population)
            
            self.location += player.y
            self.speed += player.vel_y
            self.distance += self.pipes.upper[0].x
            self.upper += self.pipes.lower[0].y + self.pipes.pipe_gap
            self.under += self.pipes.lower[0].y
            self.gap += (self.pipes.lower[0].y + (self.pipes.pipe_gap/ 2))

            self.X = np.array([
                self.location,
                self.speed,
                self.distance,
                self.upper,
                self.under,
                self.gap
            ]).T

            print(self.X)
            self.y = self.ANN.forward(self.X)

            for i, pipe in enumerate(self.pipes.upper):
                if any(player.crossed(pipe) for player in self.players):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)

            for i, player in enumerate(self.players):
                if self.y[i] == 1:
                    player.flap()
                
                survival_time = (pygame.time.get_ticks() - self.start_time) / 1000
                self.fitness[i] = survival_time + (self.score.score * 100)


            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()

            for player in self.players:
                player.tick()
                if player.collided(self.pipes, self.floor):
                    self.players.remove(player)

            self.draw_text(f"Gen: {self.gen}", 10, 10)

            elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
            self.draw_text(f"Time: {elapsed_time:.2f}s", 10, 40)

            self.draw_text(f"Max: {self.max:.2f}", 10, 70)

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

            if not self.players:
                return
