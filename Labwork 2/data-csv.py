import pandas as pd
import os

class GT:
    df = pd.read_csv(r'/home/long/longdata/mlmed/prac2/training_set_pixel_size_and_HC.csv')
    df_filename = df['filename']
    df_head_circumference = df['head circumference (mm)']

