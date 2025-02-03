
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import IPython
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X = pd.read_csv("train_values.csv")
X.head()

X.shape

"""To align the amount of rows cleaned from X to Y, we compared the indices before and after the removal process."""

X.info(verbose = True, show_counts=True)
initial_indices = X.index

"""Drop null colums/rows"""

X.isnull().sum()

"""Dropping columns that aren't needed for the project"""

X.drop(columns = ["admissions__sat_scores_25th_percentile_critical_reading",
                  "admissions__sat_scores_25th_percentile_math",
                  "admissions__sat_scores_25th_percentile_writing",
                  "admissions__sat_scores_75th_percentile_critical_reading",
                  "admissions__sat_scores_75th_percentile_math",
                  "admissions__sat_scores_75th_percentile_writing",
                  "student__share_firstgeneration_parents_middleschool",
                  "student__retention_rate_four_year_full_time",
                  "student__retention_rate_four_year_part_time",
                  "student__retention_rate_lt_four_year_full_time",
                  "student__retention_rate_lt_four_year_part_time",
                  "admissions__act_scores_25th_percentile_cumulative",
                  "admissions__act_scores_25th_percentile_english",
                  "admissions__act_scores_25th_percentile_math",
                  "admissions__act_scores_25th_percentile_writing",
                  "admissions__act_scores_25th_percentile_writing",
                  "admissions__act_scores_25th_percentile_writing",
                  "admissions__act_scores_25th_percentile_writing",
                  "admissions__act_scores_75th_percentile_cumulative",
                  "admissions__act_scores_75th_percentile_english",
                  "admissions__act_scores_75th_percentile_math",
                  "admissions__act_scores_75th_percentile_writing",
                  "admissions__act_scores_midpoint_cumulative",
                  "admissions__act_scores_midpoint_english",
                  "admissions__act_scores_midpoint_math",
                  "admissions__act_scores_midpoint_writing",
                  "admissions__sat_scores_average_by_ope_id",
                  "admissions__sat_scores_midpoint_critical_reading",
                  "admissions__sat_scores_midpoint_math",
                  "admissions__sat_scores_midpoint_writing",
                  "academics__program_percentage_agriculture",
                  "academics__program_percentage_architecture",
                  "academics__program_percentage_biological",
                  "academics__program_percentage_business_marketing",
                  "academics__program_percentage_communication",
                  "academics__program_percentage_communications_technology",
                  "academics__program_percentage_computer",
                  "academics__program_percentage_construction",
                  "academics__program_percentage_education",
                  "academics__program_percentage_engineering",
                  "academics__program_percentage_engineering_technology",
                  "academics__program_percentage_english",
                  "academics__program_percentage_ethnic_cultural_gender",
                  "academics__program_percentage_family_consumer_science",
                  "academics__program_percentage_health",
                  "academics__program_percentage_history",
                  "academics__program_percentage_humanities",
                  "academics__program_percentage_language",
                  "academics__program_percentage_legal",
                  "academics__program_percentage_library",
                  "academics__program_percentage_mathematics",
                  "academics__program_percentage_mechanic_repair_technology",
                  "academics__program_percentage_military",
                  "academics__program_percentage_multidiscipline",
                  "academics__program_percentage_parks_recreation_fitness",
                  "academics__program_percentage_personal_culinary",
                  "academics__program_percentage_philosophy_religious",
                  "academics__program_percentage_physical_science",
                  "academics__program_percentage_precision_production",
                  "academics__program_percentage_psychology",
                  "academics__program_percentage_public_administration_social_service",
                  "academics__program_percentage_resources",
                  "academics__program_percentage_science_technology",
                  "academics__program_percentage_security_law_enforcement",
                  "academics__program_percentage_social_science",
                  "academics__program_percentage_theology_religious_vocation",
                  "academics__program_percentage_transportation",
                  "academics__program_percentage_visual_performing",
                  "admissions__act_scores_25th_percentile_cumulative",
                  "admissions__act_scores_25th_percentile_english",
                  "admissions__act_scores_25th_percentile_math",
                  "admissions__act_scores_25th_percentile_writing",
                  "admissions__act_scores_75th_percentile_cumulative",
                  "admissions__act_scores_75th_percentile_english",
                  "admissions__act_scores_75th_percentile_math",
                  "admissions__act_scores_75th_percentile_writing",
                  "admissions__act_scores_midpoint_cumulative",
                  "admissions__act_scores_midpoint_english",
                  "admissions__act_scores_midpoint_math",
                  "admissions__act_scores_midpoint_writing",
                  "admissions__admission_rate_by_ope_id",
                  "admissions__admission_rate_overall",
                  "admissions__sat_scores_25th_percentile_critical_reading",
                  "admissions__sat_scores_25th_percentile_math",
                  "admissions__sat_scores_25th_percentile_writing",
                  "admissions__sat_scores_75th_percentile_critical_reading",
                  "admissions__sat_scores_75th_percentile_math",
                  "admissions__sat_scores_75th_percentile_writing",
                  "admissions__sat_scores_average_by_ope_id",
                  "admissions__sat_scores_average_overall",
                  "admissions__sat_scores_midpoint_critical_reading",
                  "admissions__sat_scores_midpoint_math",
                  "admissions__sat_scores_midpoint_writing",
                  "completion__completion_cohort_4yr_100nt",
                  "completion__completion_cohort_less_than_4yr_100nt",
                  "completion__completion_rate_4yr_100nt",
                  "completion__completion_rate_less_than_4yr_100nt",
                  "completion__transfer_rate_4yr_full_time",
                  "completion__transfer_rate_cohort_4yr_full_time",
                  "completion__transfer_rate_cohort_less_than_4yr_full_time",
                  "completion__transfer_rate_less_than_4yr_full_time",
                  "school__faculty_salary",
                  "school__ft_faculty_rate",
                  "school__institutional_characteristics_level",
                  "school__instructional_expenditure_per_fte",
                  "school__main_campus",
                  "school__online_only",
                  "school__ownership",
                  "cost__tuition_program_year",
                  "school__region_id",
                  "student__size",
                  "row_id"], inplace = True)

X.info(verbose = True, show_counts= True)

"""Since most of the columns have 16393 non-null values, we remove the rows with those missing values from all columns"""

X.dropna(subset=['academics__program_assoc_agriculture'], inplace=True)
X.info(verbose = True, show_counts= True)

X.drop_duplicates(inplace = True)

remaining_indices = X.index

"""Since the rest of the missing values are in the tution and demographics section, we replace null values in tuition section with the mean of all the values in the column. We then replace all null values in demographics section with 0 (standing for unknown).  


"""

#tuition

X['cost__tuition_in_state'] = X['cost__tuition_in_state'].fillna(X['cost__tuition_in_state'].mean())
X['cost__tuition_out_of_state'] = X['cost__tuition_out_of_state'].fillna(X['cost__tuition_out_of_state'].mean())
X['school__tuition_revenue_per_fte'] = X['school__tuition_revenue_per_fte'].fillna(X['school__tuition_revenue_per_fte'].mean())


# demographics
X['student__demographics_female_share'] = X['student__demographics_female_share'].fillna(0)
X['student__demographics_age_entry'] = X['student__demographics_age_entry'].fillna(0)
X['student__demographics_dependent'] = X['student__demographics_dependent'].fillna(0)
X['student__demographics_first_generation'] = X['student__demographics_first_generation'].fillna(0)
X['student__demographics_married'] = X['student__demographics_married'].fillna(0)
X['student__demographics_veteran'] = X['student__demographics_veteran'].fillna(0)
X['student__part_time_share'] = X['student__part_time_share'].fillna(0)
X['student__share_25_older'] = X['student__share_25_older'].fillna(0)
X['student__share_firstgeneration'] = X['student__share_firstgeneration'].fillna(0)
X['student__share_firstgeneration_parents_highschool'] = X['student__share_firstgeneration_parents_highschool'].fillna(0)
X['student__share_firstgeneration_parents_somecollege'] = X['student__share_firstgeneration_parents_somecollege'].fillna(0)
X['student__share_independent_students'] = X['student__share_independent_students'].fillna(0)
X['student__share_first_time_full_time'] = X['student__share_first_time_full_time'].fillna(0)

X.info(verbose = True)

"""Cleaning Y"""

y = pd.read_csv("train_labels.csv")
y.head()

y.info()

y = y[["income"]]
y.head()

removed_indices = initial_indices.difference(remaining_indices)
y = y.drop(removed_indices)

y.shape

"""#Scaling

These columns have large values, so we used Min-Max Scaling to transform the data between 0 and 1.
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X[['cost__tuition_in_state', 'cost__tuition_out_of_state', 'school__tuition_revenue_per_fte', 'student__demographics_age_entry']] = scaler.fit_transform(X[['cost__tuition_in_state', 'cost__tuition_out_of_state', 'school__tuition_revenue_per_fte', 'student__demographics_age_entry']])

"""#Label encoding for categorical variables"""

# categorical variables

#report_year
#school__degrees_awarded_highest
#school__degrees_awarded_predominant
#school__state

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
X['school__degrees_awarded_highest'] = label_encoder.fit_transform(X['school__degrees_awarded_highest'])
X['school__degrees_awarded_predominant'] = label_encoder.fit_transform(X['school__degrees_awarded_predominant'])
X['school__state'] = label_encoder.fit_transform(X['school__state'])

"""Ordinal encoding for report year"""

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X['report_year'] = ordinal_encoder.fit_transform(X[['report_year']])

#encoder= OneHotEncoder()



"""Use ```X_train_and_val``` and ```y_train_and_val``` for training and validation. For the final test set accuracy, use ```X_test``` and ```y_test```. You are free to do whatever for the train-val split (like cross-validation), but do not split train/val-test differently. That way, metrics can be evaluated on the same test data for each team (the last 20% of the dataset)."""

samples = len(X)
split_index = int(samples * 0.8)

X_train_and_val = X[:split_index]
y_train_and_val = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

X_train_and_val.shape

y_train_and_val.shape

X_test.shape

y_test.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

# Creating a Linear Regression model
model = LinearRegression()

# Perform k-fold cross-validation on the training and validation set
cv_scores = cross_val_score(model, X_train_and_val, y_train_and_val, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive and compute RMSE for each fold
rmse_scores = np.sqrt(-cv_scores)
print("Cross-validation RMSE scores:", rmse_scores)
print("Average Cross-validation RMSE:", rmse_scores.mean())

# Fitting the model to the data- optimizing parameters
model.fit(X_train_and_val, y_train_and_val)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating Model
r_squared = model.score(X_test, y_test)
print('R-squared:', r_squared)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# We can use rmse (root mean squared error) to make the number less big. All it does is square root the MSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

import matplotlib.pyplot as plt

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Prediction')
plt.title("Actual vs Predicted Earnings")
plt.xlabel("Actual Earnings (in $1,000s)")
plt.ylabel("Predicted Earnings (in $1,000s)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Earnings (in $1,000s)")
plt.ylabel("Residuals (in $1,000s)")
plt.grid(alpha=0.3)
plt.show()

# Print actual values and predictions (for the first 100 predictions)
for i in range(100):
  print(f"Predicted {y_pred[i]}, Actual {y_test.iloc[i]}")

# Get the weights (coefficients) and intercept
weights = model.coef_
intercept = model.intercept_

print("Weights (coefficients):", weights)
print("Intercept:", intercept)

"""#Random Forest"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
y_train_and_val = y_train_and_val.values.ravel()
# Create the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Perform k-fold cross-validation on the training and validation set
cv_scores_rf = cross_val_score(rf_model, X_train_and_val, y_train_and_val, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive and compute RMSE for each fold
rmse_scores_rf = np.sqrt(-cv_scores_rf)
print("Cross-validation RMSE scores (Random Forest):", rmse_scores_rf)
print("Average Cross-validation RMSE (Random Forest):", rmse_scores_rf.mean())

# Fit the model to the training data
rf_model.fit(X_train_and_val, y_train_and_val)

# Making predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Model
r_squared_rf = rf_model.score(X_test, y_test)
print('R-squared (Random Forest):', r_squared_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)
print('Mean Squared Error (Random Forest):', mse_rf)

rmse_rf = np.sqrt(mse_rf)
print('Root Mean Squared Error (Random Forest):', rmse_rf)

"""#Gradient boosting

"""

from sklearn.ensemble import GradientBoostingRegressor

# Create the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Perform k-fold cross-validation on the training and validation set
cv_scores_gb = cross_val_score(gb_model, X_train_and_val, y_train_and_val, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive and compute RMSE for each fold
rmse_scores_gb = np.sqrt(-cv_scores_gb)
print("Cross-validation RMSE scores (Gradient Boosting):", rmse_scores_gb)
print("Average Cross-validation RMSE (Gradient Boosting):", rmse_scores_gb.mean())

# Fit the model to the training data
gb_model.fit(X_train_and_val, y_train_and_val)

# Making predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate Model
r_squared_gb = gb_model.score(X_test, y_test)
print('R-squared (Gradient Boosting):', r_squared_gb)

mse_gb = mean_squared_error(y_test, y_pred_gb)
print('Mean Squared Error (Gradient Boosting):', mse_gb)

rmse_gb = np.sqrt(mse_gb)
print('Root Mean Squared Error (Gradient Boosting):', rmse_gb)

"""#Feature Importance for random forest


"""

# Get feature importances
rf_feature_importances = rf_model.feature_importances_

# Display the feature importances
feature_importance_df_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df_rf)

"""#Feature Importance for Gradient boosting"""

# Get feature importances
gb_feature_importances = gb_model.feature_importances_

# Display the feature importances
feature_importance_df_gb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb_feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df_gb)