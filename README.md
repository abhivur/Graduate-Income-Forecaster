
# Modeling Student Earnings After Higher Education

## Project Overview
This project aims to predict student earnings post-graduation using a comprehensive dataset of higher education factors. The focus is on developing and comparing multiple machine learning models, including **Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting**, to identify key factors influencing earnings and improve prediction accuracy.

---

## Table of Contents
1. [EDA and Preprocessing](#eda-and-preprocessing)
2. [Modeling](#modeling)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Key Findings](#key-findings)
5. [Conclusion](#conclusion)
6. [Technologies Used](#technologies-used)

---

## EDA and Preprocessing

### Steps Performed:
1. **Data Cleaning**:
   - Initial dataset contained **17,107 rows** and **298 columns**, many with missing values.
   - Columns with fewer than **100 real values** were removed.
   - Further columns deemed irrelevant to predicting earnings were dropped.
   - Rows with missing values in key columns were removed.

2. **Handling Missing Values**:
   - Tuition-related columns were imputed with the **mean**.
   - Demographics-related columns were imputed with **0**, representing unknown values.
   - Duplicate rows were removed.

3. **Scaling**:
   - Used **Min-Max Scaling** to transform numeric columns with large ranges into a scale between 0 and 1.

4. **Encoding**:
   - **Label Encoding** was applied to categorical variables.
   - **Ordinal Encoding** was used for year-based columns.

5. **Train-Test Split**:
   - Data was split into **80% training and validation** and **20% testing** to ensure consistent evaluation.

---

## Modeling

### Models Implemented:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

Each model was trained and evaluated using **k-fold cross-validation** and tested on a separate test set to ensure robustness.

---

## Evaluation Metrics

- **R-squared**: Measures the proportion of variance in the dependent variable that can be explained by the independent variables.
- **Mean Squared Error (MSE)**: Represents the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of the MSE, providing a more interpretable error measure in the same unit as the target variable.

### Model Performance:
| Model                  | R-squared | RMSE (Cross-Validation) | RMSE (Test Set) |
|------------------------|-----------|-------------------------|-----------------|
| Linear Regression      | 0.544     | 7.64                    | 7.60            |
| Random Forest Regressor| 0.816     | 5.02                    | 4.83            |
| Gradient Boosting      | 0.701     | 6.12                    | 5.87            |

---

## Key Findings

1. **Model Performance**:
   - The **Random Forest Regressor** outperformed both Linear Regression and Gradient Boosting, achieving the highest R-squared and lowest RMSE.
   - The **Linear Regression** model provided a baseline but struggled to capture complex relationships in the data.

2. **Feature Importance**:
   - Key features influencing earnings include:
     - **School Degree Predominance**
     - **Tuition Revenue per Student**
     - **Share of First-Time, Full-Time Students**

---

## Conclusion

The project demonstrates that advanced machine learning models like **Random Forest Regressor** can capture complex patterns in the data, leading to significantly improved predictions over simpler models like Linear Regression. Feature importance analysis provides valuable insights into factors influencing student earnings, which can guide educational policy and institutional strategies.

---

## Technologies Used

- **Python** for data analysis and machine learning
- **Pandas** and **NumPy** for data manipulation
- **Scikit-learn** for model implementation and evaluation
- **Matplotlib** and **Seaborn** for data visualization

