
### --- Common Imports Start --- ###
import pandas as pd
import numpy as np
### --- Common Imports End --- ###

### --- M.L. imports Start --- ###
from pycaret.time_series import TSForecastingExperiment
### --- M.L. imports End --- ###

def parse_fiscal_string(x):
    """
    Role: Breaks down the fiscal year string into tokens assigned to year and month.
    """
    s = str(x)
    year = int(s[:4])
    month = int(s[-2:])
    return pd.Timestamp(year=year, month=month, day=1)

def auto_Model_Selection(
    gr: list, 
    fh: int = 12, 
    target_metric = 'Balance', 
    time_metric = 'Fiscal Year Period',
    loss_metric = 'SMAPE', 
    random_state = None, 
    unique: bool = True
):
    # Containers from your logic
    results_container_Actuals = []
    results_container_Prediction = []
    instances_Container = {}
    model_Tier_Container = []

    # 1. Converting index and types
    for index, group in enumerate(gr):
        gr[index]['Fiscal Index'] = gr[index][time_metric].apply(parse_fiscal_string)
        gr[index].set_index('Fiscal Index', inplace=True)
        gr[index] = gr[index].asfreq('MS')
        
        # Ensure correct types
        gr[index][time_metric] = gr[index][time_metric].astype(int)
        gr[index][target_metric] = gr[index][target_metric].astype(int)

        # 2. Instantiating experiment objects
        instances_Container[f'gr{index}'] = TSForecastingExperiment()

    # 3. Init and Training
    for index, (name, experiment) in enumerate(instances_Container.items()):
        print(f"Group {name} init")
        
        # 7 Years of data pipeline
        experiment.setup(
            data=gr[index][[target_metric]], 
            target=target_metric, 
            fh=fh, 
            fold=3, 
            fold_strategy='rolling',
            session_id=random_state
        )
        
        _best7 = experiment.compare_models(n_select=10, sort=loss_metric)
        prd_7 = experiment.predict_model(_best7[0])
        model_Tier_Container.append(_best7)
        
        # 4. Testing, Plot and Print
        experiment.plot_model(_best7[0], plot='forecast', data_kwargs={'fh': fh})

        # 5. Conditional 5-year subset
        # In your logic, this runs if unique is False and data > 60 months
        if len(gr[index]) > 60 and unique is False:
            # Slicing from index 24 (2 years in) to get the last 5 years
            experiment.setup(
                data=gr[index][24:][[target_metric]], 
                target=target_metric, 
                fh=fh, 
                fold=3, 
                fold_strategy='rolling'
            )
            _best5 = experiment.compare_models(n_select=5, sort=loss_metric)
            prd_5 = experiment.predict_model(_best5[0])
            model_Tier_Container.append(_best5)
            experiment.plot_model(_best5[0], plot='forecast', data_kwargs={'fh': fh})
        else:
            prd_5 = np.array([0])

        # 6. Results and Calculations
        contribution_span = len(gr[index]) - fh
        ground_truth_figures = gr[index][target_metric][contribution_span:]
        
        results_container_Prediction.append(prd_7)
        results_container_Actuals.append(ground_truth_figures)

        # Output logic based on 'unique' flag
        if unique is False:
            print(f"Gr: {index}\nPredictions for 7 years {prd_7}\nPredictions for 5 years {prd_5}")
            
            # Loss expressed in millions calculation from your images
            group_contribution_2025 = gr[index][target_metric][contribution_span:].sum() / 1000000000
            print("The loss expressed in millions of euros: 7 years:{} and 5 years...{} vs real...{}".format(
                prd_7.sum() / 1000000000, 
                prd_5.sum() / 1000000000, 
                group_contribution_2025
            ))
            
    return results_container_Prediction, results_container_Actuals


if __name__ == "__main__":
    pass