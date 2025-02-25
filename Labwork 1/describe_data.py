from m_prepare_data import M_Data
from p_prepare_data import P_Data

m_data = M_Data()
p_data = P_Data()
m = m_data.m
p = p_data.p

print(m.describe())
print("-"*100)
print(p.describe())