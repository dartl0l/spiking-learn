import math


class Reward():
    
    def __init__(self, *args):
        pass
    
    def __call__(self, *args):
        cart_pos, cart_vel, pole_ang, pole_ang_vel = args
        return 1 / cart_pos + 1 / cart_vel + 1 / pole_ang + 1 / pole_ang_vel


class RewardAngle(Reward):
    
    def __init__(self, **kwargs):
        self.cart_pos_bound = kwargs['cart_pos_bound']
        self.cart_angle_bound = kwargs['cart_angle_bound']
    
    def __call__(self, *args):
        cart_pos, cart_vel, pole_ang, pole_ang_vel = args
        return self.cart_angle_bound / abs(abs(pole_ang) - self.cart_angle_bound)


class RewardAnglePos(Reward):
    
    def __init__(self, **kwargs):
        self.cart_pos_bound = kwargs['cart_pos_bound']
        self.cart_angle_bound = kwargs['cart_angle_bound']
    
    def __call__(self, *args):
        cart_pos, cart_vel, pole_ang, pole_ang_vel = args
        return self.cart_pos_bound / abs(abs(cart_pos) - self.cart_pos_bound) \
            + self.cart_angle_bound / abs(abs(pole_ang) - self.cart_angle_bound)

    
class RewardFormal(Reward):

    def __init__(self, **kwargs):
        self.model = kwargs['model']
    
    def __call__(self, *args):
#         observation = args
        with torch.no_grad():
            logit, reward = self.model(torch.tensor(args, dtype=torch.float32))
#         return reward
        p = F.softmax(logit, dim=-1)
        log_p = torch.log(p)
        action = torch.multinomial(p, 1)
        return action[:, 0].cpu().numpy()