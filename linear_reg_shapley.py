import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=== PART 1: Simple Linear Regression ===\n")

# Create a tiny, easy-to-understand dataset
data = {
    "size_m2":    [80, 85, 90, 95, 100, 110, 115, 120, 130, 140],
    "bedrooms":   [ 2,  3,  3,  4,   3,   4,   4,   5,   5,   5],
    "age_years":  [25,  8, 30, 12,  35,  10,  28,  15,   5,  20],
    "price_kEUR": [260, 290, 310, 355, 340, 390, 415, 460, 510, 550]
}

df = pd.DataFrame(data)
print("Our tiny dataset:")
print(df)
print("\n")

# Features and target
X = df[["size_m2", "bedrooms", "age_years"]]
y = df["price_kEUR"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Show what the model learned
print("Model learned these coefficients:")
print(f"Size (per m²)              : {model.coef_[0]:.2f} kEUR")
print(f"Bedrooms (per extra room)  : {model.coef_[1]:.2f} kEUR")
print(f"Age (per year)             : {model.coef_[2]:.2f} kEUR")
print("\n")

print("=== PART 2: Explaining predictions with SHAP ===\n")

# Create SHAP explainer (best for linear models)
explainer = shap.LinearExplainer(model, X)

# --- Example 1: Typical large house ---
new_house1 = pd.DataFrame({"size_m2": [125], "bedrooms": [4], "age_years": [5]})
pred1 = model.predict(new_house1)[0]
sv1 = explainer(new_house1)

print(f"House 1 → 125 m² with 4 bedrooms")
print(f"Predicted price: {pred1:.1f} kEUR")
print(f"SHAP size_m2   : {sv1.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv1.values[0][1]:+.2f} kEUR")
print(f"SHAP age  : {sv1.values[0][2]:+.2f} kEUR\n")

# --- Example 2: Atypical house (large but few bedrooms) ---
new_house2 = pd.DataFrame({"size_m2": [140], "bedrooms": [2], "age_years": [20]})
pred2 = model.predict(new_house2)[0]
sv2 = explainer(new_house2)

print(f"House 2 → 140 m² with only 2 bedrooms")
print(f"Predicted price: {pred2:.1f} kEUR")
print(f"SHAP size_m2   : {sv2.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv2.values[0][1]:+.2f} kEUR")
print(f"SHAP age  : {sv2.values[0][2]:+.2f} kEUR\n")

# Waterfall plot for the typical house (fixed version)
shap.plots.waterfall(sv1[0], max_display=5, show=False)   # sv1[0] ensures single instance

plt.title("SHAP Explanation - Typical House (125 m², 4 bedrooms)")
plt.tight_layout()   # Helps with layout
plt.show()           # This is crucial in scripts

print("=== PART 3: Calculating R² Score (Coefficient of Determination) ===\n")

# Step 1: Calculate mean of actual prices
y_mean = np.mean(y)

# Step 2: Calculate total variance (SS_total)
ss_total = np.sum((y - y_mean) ** 2)

# Step 3: Calculate explained variance (SS_explained)
y_pred = model.predict(X)
ss_explained = np.sum((y_pred - y_mean) ** 2)

# Step 4: Calculate R² using the formula from your script
r2 = ss_explained / ss_total

print(f"SS_total (Variance of actual data)    : {ss_total:.2f}")
print(f"SS_explained (Variance of predictions): {ss_explained:.2f}")
print(f"R² Score                               : {r2:.4f}")
print(f"→ The model explains {r2*100:.1f}% of the variation in house prices.\n")

# === PART 4: VIF CHECK ===
print("=== PART 4: VIF - Multicollinearity Check ===\n")

X_const = sm.add_constant(X)  # Important: add intercept

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1) 
                   for i in range(X.shape[1])]

print(vif_data.round(3))