import pandas as pd
import numpy as np
import Eighteen_calculate_idiosyncratic_beliefs as ib
import Nineteen_calculate_hidden_beliefs_index as hbi
import Twenty_HBI_quintiles as ffa

FOLDER = 'gravesReplication'    
pre_regression_data = pd.read_parquet(f'{FOLDER}/regression_ready_dataframe.parquet', FOLDER)
institution_demand_parameters = pd.read_parquet(f'{FOLDER}/institution_demand_parameters.parquet')

result = ib.compute_idiosyncratic_beliefs(pre_regression_data, institution_demand_parameters)
indexes = hbi.aggregate_hbi_to_stock(result, FOLDER,f'{FOLDER}/dynamic_manager_consideration_sets.csv')
alpha = ffa.assign_hbi_quintiles(indexes, FOLDER)
