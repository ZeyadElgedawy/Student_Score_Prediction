import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats

StudentPerformance_df = pd.read_csv('D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_1.csv')

X = StudentPerformance_df['Hours_Studied']
y = StudentPerformance_df['Exam_Score']

print(X.shape)
print(y.shape)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.2 , random_state = 42
)

scaler = StandardScaler