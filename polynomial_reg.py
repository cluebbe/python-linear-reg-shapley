import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ------------------- Step 1: Create sample data -------------------
np.random.seed(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = 0.5 * X**3 - 1.5 * X**2 + X + np.random.randn(50, 1) * 1.5
y = y.ravel()

# ------------------- Step 2: Visualize the sample data -------------------
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Sample data', alpha=0.7)
plt.title('Sample Data for Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# ------------------- Step 3: Try different polynomial degrees -------------------
degrees = [1, 2, 3, 5]
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.figure(figsize=(10, 7))
plt.scatter(X, y, color='blue', label='Sample data', alpha=0.5)

for degree in degrees:
    # Create and train the model
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)
    # Predict on test data for smooth curve
    y_pred = model.predict(X_test)
    # Predict on training data for evaluation
    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    # Plot the regression curve
    plt.plot(X_test, y_pred, label=f'Degree {degree} (R²={r2:.2f})')
    print(f"Degree {degree}: MSE={mse:.2f}, R²={r2:.3f}")

plt.title('Polynomial Regression Fits for Different Degrees')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()