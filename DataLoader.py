class DataLoader():
    def __init__(self,input_data,date_column,candidates_column,susceptible_column,unnormalized_cols):
        self.input_data=input_data
        self.date_col=date_column
        self.states_col=candidates_column
        self.susc_col=0
        self.num_candidates=len(set(input_data[input_data.columns[candidates_column]]))

        df=input_data
        #normalizing columns
        for col in unnormalized_cols:
            normalized_val=[]
            col_name=df.columns[col]
            series=df[col_name]
            for i in range(0,len(series),self.num_candidates):
                arr=series[i:i+self.num_candidates].to_numpy()
                for s in range(self.num_candidates):
                    normalized_val.append(arr[s]/sum(arr))
            df[col_name]=normalized_val


        date_col=date_column; state_col=candidates_column; susc_col=susceptible_column
        filtered_cols=[susc_col]
        for col in range(len(df.columns)):
            if col not in [date_col,state_col,susc_col]:
                filtered_cols.append(col)
        self.context_data=df.filter(df.columns[filtered_cols])