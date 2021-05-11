# VacSIM-PlugNPlay
This library is a PlugNPlay version of our novel pipeline [**VacSIM**](https://arxiv.org/abs/2009.06602 "Arxiv link"). 

# Overview
A COVID-19 vaccine is our best bet for mitigating the ongoing onslaught of the pandemic. However, vaccine is also expected to be a limited resource. An optimal allocation strategy, especially in countries with access inequities and temporal separation of hot-spots, might be an effective way of halting the disease spread. We approach this problem by proposing a novel pipeline **VacSIM** which follows reinforcement learning based approach for optimal distribution of vaccines for Covid-19. 
![VacSIM Pipeline](https://user-images.githubusercontent.com/22033852/117818061-dfcbc500-b285-11eb-8b98-cfdbb73cbfeb.jpg "VacSIM Pipeline")

# Quickstart
Clone the github repo.
```python
!git clone https://github.com/Saifahmadkhan/PlugNPlay
```
Select tenserflow version 1.x
```python
%tensorflow_version 1.x
```
Install the requirements.
```python
!pip install -r PlugNPlay/requirements1.txt
!sed 's/#//' PlugNPlay/requirements2.txt | xargs apt install
```
Import DataLoader and Pipeline classes.
```python
from PlugNPlay.DataLoader import DataLoader
from PlugNPlay.Pipeline import Pipeline
```
Load context data into a pandas dataframe.
```python
import pandas as pd
df=pd.read_csv('projections.csv')
df.head()
```
|Date|State / UT|Confirmed cases|Death Rate|Recovery Rate|Population|Susceptible|Overall Hospital Beds|Overall ICU Beds|Ventilators|Age Distribution|
|:---:|:-------:|:-------------:|:--------:|:-----------:|:--------:|:---------:|:-------------------:|:--------------:|:---------:|:--------------:|
|2020-12-01|Assam|2.188312e+05  |	0.011089 |0.954143	   |31205576	|3.098674e+07|24178	              |1209	           | 604  	   |4127732         |
|2020-12-01|Delhi|5.574192e+05	|0.011039	 |0.905071     |16787941	|1.623052e+07|	39455	            |1973	           |986        |	2343806       |
|2020-12-01|Jharkhand|1.111680e+05|0.011081|0.948757	   |32988134	|3.287697e+07|	26496	            |1325	           |662	       |4528093         |
|2020-12-01|Maharashtra|1.860920e+06|0.010928|0.906055	 |112374333 |1.105134e+08|	231739	          |11587           |5793	     |19912066        |
|2020-12-01|Nagaland   |1.210262e+04|0.010670|0.812338	 |1978502   |1.966399e+06|	2561	            |128	           |64	       |211983          |

Create DataLoader object by passing context dataframe, column number corresponding to date, column number corresponding to regions/candidates, column number corresponding to susceptible population and column numbers where context values are not normalized(sum over all the regions/candidates is not equal to 1).
```python
data_obj=DataLoader(df,date_column=0,candidates_column=1,susceptible_column=6,unnormalized_cols=[2,6,5,7,8,9,10])
```
Create Pipeline object by passing DataLoader object and model type. Model types allowed are 'DQN' and 'ACKTR'.
```python
pipeline_obj=Pipeline(data_object=data_obj,model_type='DQN')
```
Run the pipeline and get the distributions in `output.csv`.
```python
pipeline_obj.run()
```

# Citation
If you use Cardea for your research, please consider citing the following paper:

[VacSIM: Learning Effective Strategies for COVID-19 Vaccine Distribution using Reinforcement Learning](https://arxiv.org/abs/2009.06602).

Raghav Awasthi, Keerat Kaur Guliani, Saif Ahmad Khan, Aniket Vashishtha, Mehrab Singh Gill, Arshita Bhatt, Aditya Nagori, Aniket Gupta, Ponnurangam Kumaraguru, Tavpritesh Sethi.
