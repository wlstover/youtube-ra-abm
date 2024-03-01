from mesa import Agent, Model

class Video(Agent):
    def __init__(self, unique_id, model, prize, cost):
        super().__init__(unique_id, model)
        self.pos = None
        self.prize = prize
        self.cost = cost
        self.opened = False
        self.recommended = False
        self.likes = 0