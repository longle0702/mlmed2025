import pandas as pd
from sklearn.model_selection import train_test_split

class P_Data:
    pa = pd.read_csv(r'D:\USTH\\Nam Ba\\ml in med\\mlmed2025\\Labwork 1\\data\\ptbdb_abnormal.csv', header=None)
    pn = pd.read_csv(r'D:\USTH\\Nam Ba\\ml in med\\mlmed2025\\Labwork 1\\data\\ptbdb_normal.csv', header=None)

    classes = {0: 'Normal', 1: 'Abnormal'}

    p = pd.concat([pa, pn], axis=0, ignore_index=True)
    p.to_csv('ptbdb_total.csv', header=False, index=False)
    
    p_label = p.iloc[:, -1]
    p_data = p.iloc[:, :-1]

    p_train, p_test, p_train_label, p_test_label = train_test_split(p_data, p_label, test_size=0.2, random_state=42)
    p_train_label = p_train_label.map(classes)
    p_test_label = p_test_label.map(classes)

    print("Okay!")
