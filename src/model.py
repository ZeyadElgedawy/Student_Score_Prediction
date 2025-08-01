import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from scipy import stats

StudentPerformance_df = pd.read_csv('D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_1.csv')

X = StudentPerformance_df[['Hours_Studied']]
y = StudentPerformance_df['Exam_Score']

print(X.shape)
print(y.shape)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.2 , random_state = 42
)

scaler = StandardScaler()

X_train_final = scaler.fit_transform(X_train)
X_test_final = scaler.transform(X_test)

lr = LinearRegression()

lr.fit(X_train_final , y_train)

y_pred = lr.predict(X_test_final)

r2 = r2_score(y_test , y_pred)

mae = mean_absolute_error(y_test , y_pred)

rmse = np.sqrt(mean_squared_error(y_test , y_pred))


print("\nModel 1 Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual', y='Predicted', data=results_df, color='dodgerblue', s=50)
sns.lineplot(x='Actual', y='Actual', data=results_df, color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs Predicted Exam Scores for model 1')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
plt.show()

## preprocessing_and_visualization_2

X_train = pd.read_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\X_train.csv")
X_test = pd.read_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\X_test.csv")
y_train = pd.read_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\y_train.csv")
y_test = pd.read_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\y_test.csv")

lr.fit(X_train , y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test , y_pred)

mae = mean_absolute_error(y_test , y_pred)

rmse = np.sqrt(mean_squared_error(y_test , y_pred))

print("\nModel 2 Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

results2_df = pd.DataFrame({
    'Actual': y_test.squeeze(),
    'Predicted': y_pred.squeeze()
})

plt.figure(figsize=(8,6))
sns.scatterplot(x='Actual', y='Predicted', data=results2_df, color='dodgerblue', s=50)
sns.lineplot(x='Actual', y='Actual', data=results2_df, color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs Predicted Exam Scores for model 2')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
plt.show()