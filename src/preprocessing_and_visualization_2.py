import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import remove_z_outliers_multi

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

StudentPerformance_all_features_df = StudentPerformance_all_features_df.dropna()
missing_values = StudentPerformance_all_features_df.isnull().sum()
print("\nMissing values:")
print(missing_values)

StudentPerformance_numerical_df = StudentPerformance_all_features_df.select_dtypes(include=[np.number])

duplicated = StudentPerformance_all_features_df.duplicated().sum()
print("\nNumber of duplicated rows:",duplicated)

corr_matrix = StudentPerformance_numerical_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Feature Correlation Heatmap.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.boxplot(x = StudentPerformance_all_features_df['Hours_Studied'])
plt.title('Boxplot of Hours_Studied')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Hours_Studied.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.boxplot(x = StudentPerformance_all_features_df['Attendance'])
plt.title('Boxplot of Attendance')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Attendance.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.boxplot(x = StudentPerformance_all_features_df['Exam_Score'])
plt.title('Boxplot of Exam_Scored')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Exam_Score.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

StudentPerformance_all_features_df = remove_z_outliers_multi(
    StudentPerformance_all_features_df,
    ['Hours_Studied' , 'Exam_Score']
)

sns.boxplot(x = StudentPerformance_all_features_df['Hours_Studied'])
plt.title('Boxplot of Hours_Studied after removing outliers')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Hours_Studied after removing outliers.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

sns.boxplot(x = StudentPerformance_all_features_df['Exam_Score'])
plt.title('Boxplot of Exam_Scored')
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Boxplot of Exam_Score after removing outliers.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

y =  StudentPerformance_all_features_df['Exam_Score']
X = StudentPerformance_all_features_df.drop('Exam_Score' , axis = 1)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.2 , random_state = 42
)

encoder = OneHotEncoder(drop = 'first' , sparse_output = False)

X_cat_col = X_train.select_dtypes(include = 'object').columns
X_num_col = X_train.select_dtypes(include=[np.number]).columns

print(X_cat_col)
print(X_num_col)

X_train_encoded_cat = encoder.fit_transform(X_train[X_cat_col])
X_test_encoded_cat  = encoder.transform(X_test[X_cat_col])

X_train_encoded_cat = pd.DataFrame(
    X_train_encoded_cat,
    columns = encoder.get_feature_names_out(X_cat_col),
    index = X_train.index
)
X_test_encoded_cat = pd.DataFrame(
    X_test_encoded_cat,
    columns = encoder.get_feature_names_out(X_cat_col),
    index = X_test.index
)

print(X_train_encoded_cat[:5])
print(X_test_encoded_cat[:5])


scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[X_num_col])
X_test_num =  scaler.transform(X_test[X_num_col])

X_train_num = pd.DataFrame(
    X_train_num,
    columns = X_num_col,
    index = X_train.index
)
X_test_num = pd.DataFrame(
    X_test_num,
    columns = X_num_col,
    index = X_test.index
)

X_train_num.hist(figsize=(10, 5))
plt.suptitle("Training numerical features after scaling")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Training numerical features after scaling.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()

X_test_num.hist(figsize=(10, 5))
plt.suptitle("Test numerical features after scaling")
plots_save_path = r"D:\Elevvo_projects\Student_Score_Prediction\outputs\Test numerical features after scaling.png"
plt.savefig(plots_save_path, bbox_inches='tight', dpi=300)
plt.show()


X_train_final = pd.concat([X_train_num , X_train_encoded_cat] , axis = 1)
X_test_final  = pd.concat([X_test_num , X_test_encoded_cat] , axis = 1)


print("\nX_train Head:")
print(X_train_final.head)
print("\nTraining Data After Scaling:")
print(X_train_final[X_num_col].describe().round(2))


print("\nX_test Head:")
print(X_test_final.head)
print("\nTest Data After Scaling:")
print(X_test_final[X_num_col].describe().round(2))


X_train_final.to_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\X_train.csv", index=False)
X_test_final.to_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\X_test.csv", index=False)
y_train.to_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\y_train.csv", index=False)
y_test.to_csv(f"D:\Elevvo_projects\Student_Score_Prediction\data\processed\cleaned_student_data_2\y_test.csv", index=False)
