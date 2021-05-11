class StatesEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self, episodes, state_condition_array, susc_col, number_of_context_parameters, buckets):    #susceptible ratio is ratio of susceptible people in that state/total susceptible people combining all states
        #number_of_context_parameters=9
        self.observation_space = spaces.Box(np.array([0,0,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1,1]), shape=(number_of_context_parameters,), dtype = np.float32)	#continuous observation space 
        self.action_space = spaces.Discrete(buckets)	#discrete action space
        self.curr_step = 1
        self.done = False
        self.episodes = episodes
        self.states_cond = np.array(state_condition_array)
        self.susc_col=susc_col
        self.action_list = None
        self.buckets=buckets
        self.reset()
    def get_discrete_int(self, n):
        discrete_int = int(n)
        return discrete_int

    def reset(self):
        self.curr_step = 1
        self.done = False
        #self.states_cond =  np.array([])
        return copy.deepcopy(self.states_cond)
        
    def step(self, action):
        self.action_list = action
        reward = self.get_reward()
        # print("susc ratio",self.states_cond[0])
        # print("action",action,action/self.buckets)
        # print("reward",reward)
        # increment episode
        if self.curr_step == self.episodes:
            self.done=True
        else:
            self.done=False
            self.curr_step+=1
        
        return self.states_cond, reward, self.done, {'action_list': self.action_list, 'distribution':(self.action_list*(1/self.buckets))+(0.5/self.buckets), 'episode_number': self.curr_step}
    
    def get_reward(self):
        reward = 0              
        A=(self.action_list*(1/self.buckets))
        S=self.states_cond[self.susc_col]                                                    
        reward = (100 * math.exp((-(A-S)**2)/0.0001))
        return reward 

    def close(self):
        pass