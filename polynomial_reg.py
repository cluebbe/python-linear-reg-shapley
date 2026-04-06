import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# ------------------- Step 1: Create sample data -------------------
# Generate synthetic data to demonstrate polynomial regression
# We create 50 data points between -3 and 3 for our input feature X
np.random.seed(42)  # Set random seed for reproducible results
X = np.linspace(-3, 3, 50).reshape(-1, 1)          # Input feature: 50 points evenly spaced from -3 to 3, reshaped to column vector

# Create target values y based on a cubic polynomial relationship: y = 0.5*x³ - 1.5*x² + x + noise
# This simulates real-world data where the relationship isn't linear
y = 0.5 * X**3 - 1.5 * X**2 + X + np.random.randn(50, 1) * 1.5

# Flatten y from 2D array to 1D array (required by scikit-learn)
y = y.ravel()

# ------------------- Step 2: Visualize the sample data -------------------
# Plot only the generated training data so you can inspect the input-output pairs
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Sample data', alpha=0.7)
plt.title('Sample Data for Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# ------------------- Step 3: Create and train the polynomial regression model -------------------
# Polynomial regression with scikit-learn works by transforming features into polynomial terms
# then applying linear regression. Here we use degree 3, so we'll have x, x², and x³ terms
degree = 3

# Create a pipeline that first transforms features to polynomial, then applies linear regression
# PolynomialFeatures(degree=degree, include_bias=False): Creates polynomial features up to degree 3
#   - For input x, creates [x, x², x³] (bias/intercept handled by LinearRegression)
#   - include_bias=False prevents adding a constant term (1) since LinearRegression adds it
# LinearRegression(): Fits a linear model to the transformed polynomial features
model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression()
)

# Train (fit) the model on our data X and y
model.fit(X, y)

# ------------------- Step 4: Make predictions on new data -------------------
# Create a denser set of test points for smooth plotting (200 points instead of 50)
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)

# Use the trained model to predict y values for the test points
# The pipeline automatically transforms X_test to polynomial features, then predicts
y_pred = model.predict(X_test)

# ------------------- Step 5: Visualize the regression result -------------------
# Create a figure for plotting the model prediction against the original data
plt.figure(figsize=(8, 6))

# Plot the original training data as blue scatter points
plt.scatter(X, y, color='blue', label='Actual data', alpha=0.7)

# Plot the model's predictions as a red line
plt.plot(X_test, y_pred, color='red', linewidth=2, label=f'Polynomial Regression (degree={degree})')

# Add plot labels and formatting
plt.title('Polynomial Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# ------------------- Step 6: Examine the learned coefficients -------------------
# Access the individual steps of the pipeline to inspect the results
poly_features = model.named_steps['polynomialfeatures']  # The polynomial transformation step
linear_model = model.named_steps['linearregression']     # The linear regression step

# Print the coefficients learned by the linear regression
# These correspond to the polynomial terms: [coefficient for x³, coefficient for x², coefficient for x]
print("Polynomial coefficients (from highest degree to lowest):")
print(linear_model.coef_)
print(f"Intercept: {linear_model.intercept_:.4f}")