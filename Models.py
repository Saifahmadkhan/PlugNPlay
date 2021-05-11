from stable_baselines import DQN,ACKTR
class DQN_model():
    def __init__(self, policy='MlpPolicy', learning_rate=1e-3, prioritized_replay=True, verbose=1, exploration_fraction=0.2):
        self.policy=policy
        self.learning_rate=learning_rate
        self.prioritized_replay=prioritized_replay
        self.verbose=verbose
        self.exploration_fraction=exploration_fraction
    def build_model(self,env):
        return DQN(self.policy,env,learning_rate=self.learning_rate,prioritized_replay=self.prioritized_replay,verbose=self.verbose,exploration_fraction=self.exploration_fraction)

class ACKTR_model():
    def __init__(self,policy='MlpPolicy', verbose=1,n_steps=1, seed=1, learning_rate=1e-3):
        self.policy=policy
        self.verbose=verbose
        self.n_steps=n_steps
        self.seed=seed
        self.learning_rate=learning_rate
    def build_model(self,env):
        return ACKTR(self.policy,env,verbose=self.verbose,n_steps=self.n_steps,seed=self.seed,learning_rate=self.learning_rate)
