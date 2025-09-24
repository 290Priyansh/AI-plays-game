import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load labeled data
df = pd.read_csv("labeled_data.csv")

# Separate features and labels
X = df[["dx", "dy", "vel_y","spring"]]       # input state
y = df["action"]                    # action to take

# Encode actions into numbers (for classification)
action_map = {"LEFT": 0, "RIGHT": 1, "NONE": 2}
reverse_map = {0: "LEFT", 1: "RIGHT", 2: "NONE"}
y = y.map(action_map)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
with open("supervised_model.pkl", "wb") as f:
    pickle.dump((model, reverse_map), f)
print("✅ Model saved to 'supervised_model.pkl'")
