from .output import Output

class AdaptiveOutput(Output):
    def __init__(self, init_params=None, init_t=None):
        self.init_params = init_params
        self.init_t = init_t

        self.params = init_params
        self.t = init_t

    def set_init(self, init_params, init_t):
        self.init_params = init_params
        self.init_t = init_t

    def update_params(self, x, u, t):
        pass

    def reset(self):
        self.params = self.init_params
        self.t = self.init_t
