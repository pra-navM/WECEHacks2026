import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text  # Changed to Regressor
from stable_baselines3 import PPO

# 1. Load Model
model = PPO.load("rl/checkpoints/scheduler_pro_v1.zip")

# 2. Setup Feature Names
feature_names = [f"f{i}" for i in range(134)]

# 3. Generate samples
num_samples = 50000 
observations = np.random.uniform(-1, 1, size=(num_samples, 134)).astype(np.float32)

print("🎲 Predicting weights with PPO...")
# For the Governor, actions are continuous weights (Box space)
actions, _ = model.predict(observations, deterministic=True)

# 4. Train the "Mimic" (Regressor)
# We use a Regressor because actions are continuous floats
reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=20)
print("🧠 Distilling weights into tree logic...")
reg.fit(observations, actions)

# 5. Export
# Accuracy for regressors is measured via R^2 score (1.0 is perfect)
score = reg.score(observations, actions)
tree_rules = export_text(reg, feature_names=feature_names)

with open("distilled_logic.txt", "w") as f:
    f.write(f"Policy Fidelity (R^2 Score): {score:.4f}\n")
    f.write("Note: Output values represent [Weight_Length, Weight_Affinity, Weight_Thermal]\n")
    f.write("="*30 + "\n")
    f.write(tree_rules)

print(f"🧠 Policy Distilled! Fidelity: {score:.4f}")