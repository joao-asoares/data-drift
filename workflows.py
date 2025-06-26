import pandas as pd
import numpy as np
from studd.studd_batch import STUDD
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RF

from skmultiflow.drift_detection.page_hinkley import PageHinkley as PHT

def plot_results(errorTeacher, errorStud, errorbetweenboth, std_alarms, step, title="STUDD"):
    import matplotlib.pyplot as plt
    y = range(1000, 1000 + len(errorTeacher))
    plt.clf()

    plt.plot(y, errorStud, label='Student')
    plt.plot(y, errorTeacher, label='Teacher')
    plt.plot(y, errorbetweenboth, label='Between')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    plt.title(title)

    for i in std_alarms:
        plt.axvline(x=i, color='black', linestyle='--', label=f"drift_{i}")

    plt.savefig(f'error{step}.png')
    plt.show()

def Workflow(X, y, delta, window_size):
    ucdd = STUDD(X=X, y=y, n_train=window_size)

    ucdd.initial_fit(model=RF(), std_model=RF())

   
    print("Detecting change with STUDD")
    RES_STUDD = ucdd.drift_detection_std(datastream_=ucdd.datastream,
                                        model_=ucdd.base_model,
                                        std_model_=ucdd.student_model,
                                        n_train_=ucdd.n_train,
                                        n_samples=window_size,
                                        delta=delta / 2,
                                        upd_model=True,
                                        upd_std_model=True,
                                        detector=PHT)



    training_info = ucdd.init_training_data

    results = {
               "STUDD": RES_STUDD,
                }
    perf_kpp = dict()
    perf_acc = dict()
    nupdates = dict()
    pointsbought = dict()
    for m in results:
        x = results[m]
        perf_acc_i = metrics.accuracy_score(y_true=x["preds"]["y"],
                                            y_pred=x["preds"]["y_hat"])

        perf_m = metrics.cohen_kappa_score(y1=x["preds"]["y"],
                                           y2=x["preds"]["y_hat"])

        pointsbought[m] = x["samples_used"]
        nupdates[m] = x["n_updates"]

        perf_kpp[m] = perf_m
        perf_acc[m] = perf_acc_i

    perf_kpp = pd.DataFrame(perf_kpp.items())
    perf_acc = pd.DataFrame(perf_acc.items())

    perf = pd.concat([perf_kpp.reset_index(drop=True), perf_acc], axis=1)

    perf.columns = ['Method', 'Kappa', 'rm', 'Acc']
    # plot_results(perf["Kappa"].values, perf["Acc"].values,
    #              perf["Kappa"].values, ucdd.std_alarms, window_size, title="STUDD")
    
    return perf, pointsbought, nupdates, training_info, results

