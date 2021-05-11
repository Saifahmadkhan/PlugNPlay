class Pipeline():
    def __init__(self,data_object,model_type="DQN"):
        if model_type not in ['ACKTR','DQN']:
            raise Exception("Invalid model type {}. model_type must be from {}.".format(model_type,['ACKTR','DQN']))
        self.data=data_object
        self.model_type=model_type
        
    def run(self,iterations=int(1e4),buckets=1000,total_vials=1000000):
        df=self.data.context_data
        reward_list=[]; distribution_list=[]
        num_candidates=self.data.num_candidates
        susc_col=self.data.susc_col

        for i in range(0,len(df),num_candidates):
            actions=[]
            arr=df[i:i+num_candidates].to_numpy()
            for s in range(num_candidates):
                context=arr[s] 
                env=StatesEnv(episodes=1, state_condition_array=context, susc_col=susc_col, number_of_context_parameters=len(context),buckets=buckets)
                if self.model_type=="DQN":
                    model=DQN_model()
                elif self.model_type=="ACKTR":
                    model=ACKTR_model()
                model=model.build_model(env)
                model.learn(total_timesteps=iterations, log_interval=iterations/5)
                obs = env.reset()
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                actions.append(action)
                reward_list.append(reward)
                print("prediction: action,reward",action,reward)
                #return
            for s in range(num_candidates):
                fraction=actions[s]/sum(actions)
                distribution_list.append(fraction*total_vials)

        self.output(distribution_list,reward_list)

    def output(self,distribution_list,reward_list):
        df=self.data.input_data
        df['Distributions']=distribution_list
        df['Reward']=reward_list
        df.to_csv('output.csv')