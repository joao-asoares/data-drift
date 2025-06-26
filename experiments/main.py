import pickle
import os

from workflows import Workflow
import pandas as pd


# Load the dataset from a CSV file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data', 'insects.csv')
df = pd.read_csv(csv_path)

# Rename the columns of the dataframe to be sequential integers, 
# with the last column named "target".
column = []
for i in range(df.shape[1] - 1):
    column.append(i)
column.append("target")
df.columns = column

delta = 0.02

results = dict()
for i in range(1):
    
    y = df.target.values
    X = df.drop(['target'], axis=1)
   

    W = 1000
    
    predictions, detections, train_size, training_info, results_comp = \
        Workflow(X=X, y=y,delta=delta,window_size=W)
    
    ds_results = \
        dict(predictions=predictions,
             detections=detections,
             n_updates=train_size,
             data_size=len(y),
             training_info=training_info,
             results_comp=results_comp)
    
    results[f"experiment_{i}"] = ds_results
    
    dump_path = os.path.join(base_dir, 'data', 'studd_experiments.pkl')
    with open(dump_path, 'wb') as fp:
        pickle.dump(results, fp)
