import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    'd:/Sumanyu/Relearning Python/4) Mini Projects/1) Different variables and their impact on diabetes outcomes/diabetes.csv')
df['Pregnancies'] = pd.to_numeric(df['Pregnancies'])
df['Glucose'] = pd.to_numeric(df['Glucose'])
df['BloodPressure'] = pd.to_numeric(df['BloodPressure'])
df['SkinThickness'] = pd.to_numeric(df['SkinThickness'])
df['Insulin'] = pd.to_numeric(df['Insulin'])
df['BMI'] = pd.to_numeric(df['BMI'])
df['DiabetesPedigreeFunction'] = pd.to_numeric(df['DiabetesPedigreeFunction'])
df['Age'] = pd.to_numeric(df['Age'])
df['Outcome'] = pd.to_numeric(df['Outcome'])
df = df[(df['Insulin'] != 0) & (df['Glucose'] != 0)]
df = df.dropna(subset=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

print(f'{df.shape}\n')
print(df.describe())
print(f"\n{df['Glucose'].corr(df['DiabetesPedigreeFunction'])}\n")
print(f"\n{df['Glucose'].corr(df['BloodPressure'])}\n")
print(f"\n{df['Glucose'].corr(df['Insulin'])}\n")
print(f'{(df == 0).sum()}')

x_data = df['Glucose']
y_data = df['Insulin']
mean_insulin = np.mean(y_data)
ss_tot = np.sum((y_data-mean_insulin)**2)
print(f'\nSS_tot= {ss_tot}\n')

x_array = np.linspace(50, 200, 200)

x_data_ndiabetic = df[df['Outcome'] == 0]['Glucose']
y_data_ndiabetic = df[df['Outcome'] == 0]['Insulin']
mean_ndiabetic_insulin = np.mean(y_data_ndiabetic)
ss_tot_ndiabetic = np.sum((y_data_ndiabetic-mean_ndiabetic_insulin)**2)

x_data_diabetic = df[df['Outcome'] != 0]['Glucose']
y_data_diabetic = df[df['Outcome'] != 0]['Insulin']
mean_diabetic_insulin = np.mean(y_data_diabetic)
ss_tot_diabetic = np.sum((y_data_diabetic-mean_diabetic_insulin)**2)

coeffs_ndiabetic = np.polyfit(x_data_ndiabetic, y_data_ndiabetic, 1)
y_ndiabetic_predicted = np.polyval(coeffs_ndiabetic, x_data_ndiabetic)
y_ndiabetic_model = np.polyval(coeffs_ndiabetic, x_array)
y_ndiabetic_residuals = y_data_ndiabetic-y_ndiabetic_predicted
sse_ndiabetic = np.sum(y_ndiabetic_residuals**2)
r_sqr_ndiabetic = 1-(sse_ndiabetic/ss_tot_ndiabetic)

coeffs_diabetic = np.polyfit(x_data_diabetic, y_data_diabetic, 1)
y_diabetic_predicted = np.polyval(coeffs_diabetic, x_data_diabetic)
y_diabetic_model = np.polyval(coeffs_diabetic, x_array)
y_diabetic_residuals = y_data_diabetic-y_diabetic_predicted
sse_diabetic = np.sum(y_diabetic_residuals**2)
r_sqr_diabetic = 1-(sse_diabetic/ss_tot_diabetic)

plt.figure(figsize=(12, 8))
plt.title('Blood Glucose Concentration VS Blood Pressure')
plt.scatter(x_data, df['BloodPressure'], marker='x')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.ylabel('Blood Pressure (mm Hg)')
plt.grid(True, linestyle=':')

plt.figure(figsize=(12, 8))
plt.title('Blood Glucose Concentration VS Skin Thickness')
plt.scatter(x_data, df['SkinThickness'], marker='x')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.ylabel('Skin Thickness (mm)')
plt.grid(True, linestyle=':')

plt.figure(figsize=(12, 8))
plt.title('Blood Glucose Concentration VS Insulin Concentration')
plt.scatter(x_data, y_data, marker='x')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.ylabel('Insulin Concentration/(mu U/ml)')
plt.grid(True, linestyle=':')

plt.show()


plt.figure(figsize=(12, 8))
plt.suptitle('Sub-divided Residual Plot')

plt.subplot(2, 1, 1)
plt.scatter(x_data_diabetic, y_diabetic_residuals,
            marker='x', color='red', alpha=0.5)
plt.title('Diabetic Residuals')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.ylabel('Insulin Concentration/(mu U/ml)')
plt.axhline(0, linestyle='--', color='green')
plt.grid(True, linestyle=':')

plt.subplot(2, 1, 2)
plt.scatter(x_data_ndiabetic, y_ndiabetic_residuals,
            marker='x', color='blue', alpha=0.5)
plt.title('Non-diabetic Residuals')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.ylabel('Insulin Concentration/(mu U/ml)')
plt.axhline(0, linestyle='--', color='green')
plt.grid(True, linestyle=':')
plt.tight_layout()

plt.figure(figsize=(12, 8))
plt.scatter(x_data_ndiabetic, y_data_ndiabetic,
            marker='x', color='blue', label='Non-Dabetic', alpha=0.5)
plt.plot(x_array, y_ndiabetic_model, color='blue', label='Non-Diabetic')
plt.scatter(x_data_diabetic, y_data_diabetic,
            marker='x', color='red', label='Diabetic', alpha=0.5)
plt.plot(x_array, y_diabetic_model, color='red', label='Diabetic')
plt.ylabel('Insulin Concentration/(mu U/ml)')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.legend()
plt.grid(True, linestyle=':')

coeffs_lin = np.polyfit(x_data, y_data, 1)
coeffs_quad = np.polyfit(x_data, y_data, 2)
y_lin = np.polyval(coeffs_lin, x_array)
y_quad = np.polyval(coeffs_quad, x_array)

lin_y_predicted = np.polyval(coeffs_lin, x_data)
lin_residuals = y_data-lin_y_predicted
sse_lin = np.sum(lin_residuals**2)
r_sqr_lin = 1-(sse_lin/ss_tot)

quad_y_predicted = np.polyval(coeffs_quad, x_data)
quad_residuals = y_data-quad_y_predicted
sse_quad = np.sum(quad_residuals**2)
r_sqr_quad = 1-(sse_quad/ss_tot)

ln_x = np.log(x_data)
ln_y = np.log(y_data)
ln_x_array = np.linspace(4, 5.5, 200)
power_rule_coeffs = np.polyfit(ln_x, ln_y, 1)
power_rule_output = np.polyval(power_rule_coeffs, ln_x_array)
y_pow = np.exp(power_rule_output)
power_rule_predicted = np.polyval(power_rule_coeffs, ln_x)
exp_predicted = np.exp(power_rule_predicted)
pow_residuals = y_data-exp_predicted
sse_pow = np.sum(pow_residuals**2)
r_sqr_pow = 1-(sse_pow/ss_tot)

print(f'SSE (linear model): {sse_lin}')
print(f'R² (linear model): {r_sqr_lin}\n')

print(f'SSE (quadratic model): {sse_quad}')
print(f'R² (quadratic model): {r_sqr_quad}\n')

print(f'SSE (power rule model): {sse_pow}')
print(f'R² (power rule model): {r_sqr_pow}\n')

print(f'\nSSE (diabetic-quad model): {sse_diabetic}')
print(f'R² (diabetic-quad model): {r_sqr_diabetic}\n')

print(f'SSE (non-diabetic-quad model): {sse_ndiabetic}')
print(f'R² (non-diabetic-quad model): {r_sqr_ndiabetic}\n')

plt.figure(figsize=(12, 8))

plt.title('Comparison of Linear, Quadratic and Power-law model againt data')
plt.scatter(x_data, y_data, marker='x', color='blue', label='Data', alpha=0.5)
plt.plot(x_array, y_quad, label='Quadratic Model', color='orange')
plt.plot(x_array, y_lin, label='Linear Model', color='green')
plt.plot(x_array, y_pow, label='Power Rule Model', color='red')
plt.ylabel('Insulin Concentration/(mu U/ml)')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.grid(True, linestyle=':')
plt.legend()

plt.figure(figsize=(12, 8))

plt.title('Logarithmic linearistion of Power-law model')
plt.scatter(ln_x, ln_y, marker='x', color='blue',
            label='Natural log of data', alpha=0.5)
plt.plot(ln_x_array, power_rule_output,
         label='Power-Law Model', color='orange')
plt.ylabel('ln(Insulin Concentration/(mu U/ml))')
plt.xlabel('ln(Blood Glucose Concentration/(mg/dl))')
plt.grid(True, linestyle=':')
plt.legend()


plt.figure(figsize=(12, 8))
plt.suptitle('Residuals')

plt.subplot(3, 1, 1)
plt.title('Linear Residuals')
plt.scatter(x_data, lin_residuals, marker='x', color='blue', alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.ylabel('Residuals')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.grid(True, linestyle=':')

plt.subplot(3, 1, 2)
plt.title('Quadratic Residuals')
plt.scatter(x_data, quad_residuals, marker='x', color='blue', alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.ylabel('Residuals')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.grid(True, linestyle=':')

plt.subplot(3, 1, 3)
plt.title('Power-law Residuals')
plt.scatter(x_data, pow_residuals, marker='x', color='blue', alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.ylabel('Residuals')
plt.xlabel('Blood Glucose Concentration/(mg/dl)')
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
