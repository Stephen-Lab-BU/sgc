
from typing_extensions import Protocol

# Utility Functions - Should be moved later to alternative file
####################
class ParameterSet(Protocol):
    pass



class OptimResult():
    def __init__(self, zs_est, hess, track_zs=None, track_cost=None, track_grad=None, track_hess=None):
        self.zs_est = zs_est
        self.hess = hess
        self.track_zs = track_zs
        self.track_cost = track_cost
        self.track_grad = track_grad
        self.track_hess = track_hess

class OptimResultReal():
    def __init__(self, vs_est, hess_real):
        self.vs_est = vs_est
        self.hess_real = hess_real