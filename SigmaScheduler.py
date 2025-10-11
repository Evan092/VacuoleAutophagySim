import math

class ScheduledSigma:
    def __init__(self, sigma_start=0.45, sigma_end=0.0, T_max=60, mode='cosine', epoch=0):
        self.sigma_start = sigma_start
        self.sigma_end   = sigma_end
        self.T_max       = T_max
        self.mode        = mode
        self.epoch       = epoch

    @property
    def value(self):
        if self.mode == 'linear':
            # sigma(t) = sigma_start - (sigma_start - sigma_end) * (epoch/T_max)
            return self.sigma_start - (self.sigma_start - self.sigma_end) * (self.epoch / self.T_max)
        else:
            # cosine: sigma(t) = sigma_end + 0.5*(sigma_start - sigma_end)*(1 + cos(pi * epoch / T_max))
            return self.sigma_end + 0.5 * (self.sigma_start - self.sigma_end) * (1 + math.cos(math.pi * self.epoch / self.T_max))

    def step(self):
        self.epoch = min(self.epoch + 1, self.T_max)