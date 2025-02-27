from m_prepare_data import M_Data
from p_prepare_data import P_Data

m_data = M_Data()
p_data = P_Data()
m = m_data.m
p = p_data.p

print('MIT-BIH Arrhythmia Database')
print(m.info())
print("-"*100)
print('PTB Diagnostic ECG Database')
print(p.info())