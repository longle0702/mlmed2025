import pandas as pd
class M_Data:
    m_train = pd.read_csv(r'D:\USTH\\Nam Ba\\ml in med\\mlmed2025\\Labwork 1\\data\\mitbih_train.csv', header=None)
    m_test = pd.read_csv(r'D:\USTH\\Nam Ba\\ml in med\\mlmed2025\\Labwork 1\\data\\mitbih_test.csv', header=None)

    classes = {0: "Normal Beats", 1: "Supraventricular Ectopy Beats", 2: "Ventricular Ectopy Beats", 3: "Fusion Beats", 4: "Unclassifiable Beats"}
    m = pd.concat([m_train, m_test], axis=0, ignore_index=True)
    m.to_csv('mitbih_total.csv', header=False, index=False)
    m_data = m.iloc[:, :-1]
    m_label = m.iloc[:, -1].map(classes)

    m_train_label = m_train.iloc[:, -1].map(classes)
    m_test_label = m_test.iloc[:, -1].map(classes)
    m_train_data = m_train.iloc[:, :-1]
    m_test_data = m_test.iloc[:, :-1]

    print("Okay!")





