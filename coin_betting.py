import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
import copy

# game rules
p = 0.6  # probability of winning +1 (heads)
q = 1-p  # probability of losing -1 (tails)

#----------------------------------------------------


class Gambler:
    '''
    A gambler playing multiple games independently in parallel
    according to the specified policy. During each round, a possibly
    biased coin is tossed and the gambler wins if the toss is heads / +1.
    '''
    def __init__(self, initial_wealth: np.ndarray, policy: Callable):
        self.wealth = initial_wealth
        self.policy = policy

    def play(self):
        '''
        Simulates one round of play.
        '''
        B = len(self.wealth)
        bet = self.policy(self.wealth)  # (B,)
        outcome = np.random.binomial(size=B, n=1, p=p)
        outcome[outcome == 0] = -1
        self.wealth += outcome*bet
        self.wealth[self.wealth < 0] = 0
        return None

    def get_wealth(self):
        '''
        returns a copy of the current state of wealth
        '''
        return copy.deepcopy(self.wealth)


class FixedFractionPolicy:
    '''
    Policy that bets some fraction of the current wealth during each round.
    '''
    def __init__(self, f: float):
        assert f >= 0 and f <= 1, "fraction f must be in [0,1]"
        self.f = f

    def bet(self, current_wealth):
        '''Returns bet made by policy'''
        return self.f * current_wealth

    def __call__(self, current_wealth):
        return self.bet(current_wealth)



#-----------------------------------------------------

# policies
all_or_nothing_policy = FixedFractionPolicy(f=1.0)
kelly_policy = FixedFractionPolicy(f=p-q)
half_policy = FixedFractionPolicy(f=0.5)

# gamblers
num_gamblers = 50
num_rounds = 1000
wealth = 100

gamblers = {
    "all_or_nothing": Gambler(wealth * np.ones((num_gamblers,)), all_or_nothing_policy),
    "kelly": Gambler(wealth * np.ones((num_gamblers,)), kelly_policy),
    "halving": Gambler(wealth * np.ones((num_gamblers,)), half_policy)
}

keys = gamblers.keys()
# wealth processes

wealth_processes = {key: [gamblers[key].get_wealth()] for key in keys}

for ii in range(1, num_rounds):
    for key in keys:
        gamblers[key].play()
        wealth_processes[key].append(gamblers[key].get_wealth())

# plotting
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']
for i, key in enumerate(keys):
    wp = np.asarray(wealth_processes[key])  # (T, B)
    ax.plot(wp[:, 0], c=colors[i], alpha=0.6, label=key)
    ax.plot(wp[:, 1:], c=colors[i], alpha=0.6)
plt.yscale("symlog")
ax.set_title(f'Fixed fraction gambling strategies: {num_gamblers} parallel games')
ax.set_ylabel("Gambler's wealth")
ax.set_xlabel("Round")
plt.legend()
plt.savefig("gambling_strategies.png", dpi=300)
plt.close()
