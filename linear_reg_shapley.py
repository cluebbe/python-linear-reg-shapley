import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt

print("=== PART 1: Simple Linear Regression ===\n")

# Create a tiny, easy-to-understand dataset
data = {
    "size_m2":     [80, 100, 120, 90, 110, 130],
    "bedrooms":    [2,   3,   3,   2,   3,   4],
    "price_kEUR":  [250, 320, 380, 270, 340, 410]
}

df = pd.DataFrame(data)
print("Our tiny dataset:")
print(df)
print("\n")

# Separate features (X) and target (y)
X = df[["size_m2", "bedrooms"]]
y = df["price_kEUR"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Show what the model learned
print("Model learned these coefficients:")
print(f"Intercept (base price)     : {model.intercept_:.2f} kEUR")
print(f"Size (per m²)              : {model.coef_[0]:.2f} kEUR")
print(f"Bedrooms (per extra room)  : {model.coef_[1]:.2f} kEUR")
print("\n")

print("=== PART 2: Explaining predictions with SHAP ===\n")

# Create SHAP explainer (best for linear models)
explainer = shap.LinearExplainer(model, X)

# --- Example 1: Typical large house ---
new_house1 = pd.DataFrame({"size_m2": [125], "bedrooms": [4]})
pred1 = model.predict(new_house1)[0]
sv1 = explainer(new_house1)

print(f"House 1 → 125 m² with 4 bedrooms")
print(f"Predicted price: {pred1:.1f} kEUR")
print(f"SHAP size_m2   : {sv1.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv1.values[0][1]:+.2f} kEUR\n")

# --- Example 2: Atypical house (large but few bedrooms) ---
new_house2 = pd.DataFrame({"size_m2": [140], "bedrooms": [2]})
pred2 = model.predict(new_house2)[0]
sv2 = explainer(new_house2)

print(f"House 2 → 140 m² with only 2 bedrooms")
print(f"Predicted price: {pred2:.1f} kEUR")
print(f"SHAP size_m2   : {sv2.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv2.values[0][1]:+.2f} kEUR\n")

# Waterfall plot for the typical house (fixed version)
shap.plots.waterfall(sv1[0], max_display=5, show=False)   # sv1[0] ensures single instance

plt.title("SHAP Explanation - Typical House (125 m², 4 bedrooms)")
plt.tight_layout()   # Helps with layout
plt.show()           # This is crucial in scripts