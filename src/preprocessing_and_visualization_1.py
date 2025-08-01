

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

StudentPerformance_all_features_df = pd.read_csv('D:\Elevvo_projects\Student_Score_Prediction\data/raw\StudentPerformanceFactors.csv')

print("Data Frame Info:")
print(StudentPerformance_all_features_df.info())

print("\nData Frame Description (Numerical Columns):")
print(StudentPerformance_all_features_df.describe())

print("\nData Frame Description (Categorical Columns):")
print(StudentPerformance_all_features_df.describe(include="object"))

missing_values = StudentPerformance_all_features_df.isnull().sum()
print("\nMissing values:")
print(missing_values)

duplicated = StudentPerformance_all_features_df.duplicated().sum()
print("\nNumber of duplicated rows:",duplicated)


sns.boxplot(x = StudentPerformance_all_features_df['Hours_Studied'])
plt.title('Boxplot of Hours Studied')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Hours Studied.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.boxplot(x = StudentPerformance_all_features_df['Exam_Score'])
plt.title('Boxplot of Exam Score')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Exam Score.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

z_scores = stats.zscore(StudentPerformance_all_features_df[['Hours_Studied','Exam_Score']])
abs_z = abs(z_scores)
StudentPerformance_all_features_df = StudentPerformance_all_features_df[(abs_z < 3).all(axis = 1)]

our_features = ['Hours_Studied','Exam_Score']
StudentPerformance_df = StudentPerformance_all_features_df[our_features]

print(StudentPerformance_df.describe())
print(StudentPerformance_df.head())

sns.histplot(StudentPerformance_df['Hours_Studied'], kde = True)
plt.title("Distribution of Hours Studied")
plt.xlabel("Hours Studied")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Distribution of Hours Studied.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.histplot(StudentPerformance_df['Exam_Score'], kde = True)
plt.title("Distribution of Exam Score")
plt.xlabel("Exam Score")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Distribution of Exam Score.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.scatterplot(x = 'Hours_Studied' , y = 'Exam_Score' , data = StudentPerformance_df)
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.xlabel("Exam Score")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Study Hours vs Exam Score.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.regplot(x = 'Hours_Studied' , y = 'Exam_Score' , data = StudentPerformance_df , line_kws={"color":"red"})
plt.title("Linear Relationship: Hours Studied vs Final Score")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Linear Relationship: Hours Studied vs Final Score.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

StudentPerformance_df.to_csv("D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_1.csv", index=False)

