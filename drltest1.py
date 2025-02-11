import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st

# Define the neural network for deep reinforcement learning
class DRLModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DRLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Simulate data for training
def simulate_data(num_samples=1000):
    X = np.random.rand(num_samples, 5)  # 5 input features: raw materials, labor, etc.
    y = (X[:, 0] * 50 + X[:, 1] * 30 + X[:, 2] * 20 - X[:, 3] * 10 + X[:, 4] * -15).clip(0, 100).astype(np.float32)
    return X.astype(np.float32), y

# Train the DRL model
def train_model(model, optimizer, criterion, X_train, y_train, epochs=1000):
    model.train()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train)
        targets = torch.tensor(y_train).unsqueeze(1)  # Add dimension for regression output

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Predict using the trained model
def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        inputs_tensor = torch.tensor(inputs).unsqueeze(0)  # Add batch dimension
        outputs = model(inputs_tensor)
    return outputs.item()

# Save the trained model to a file
def save_model(model, path="models/drl_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create the directory if it doesn't exist
    torch.save(model.state_dict(), path)

# Load the trained model from a file
def load_model(path="models/drl_model.pth", input_dim=5, output_dim=1):
    model = DRLModel(input_dim=input_dim, output_dim=output_dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation mode
    else:
        raise FileNotFoundError(f"Model file not found at {path}")
    return model

# Streamlit UI
st.title("Deep Reinforcement Learning: Labor Resource Optimization")

# User input sliders for features
raw_material_1 = st.slider("Raw Material 1", min_value=0.0, max_value=100.0, value=50.0)
raw_material_2 = st.slider("Raw Material 2", min_value=0.0, max_value=100.0, value=50.0)
no_of_labor = st.slider("Number of Labor", min_value=1.0, max_value=500.0, value=50.0)
no_of_days_to_execute = st.slider("Number of Days to Execute", min_value=1.0, max_value=30.0, value=10.0)
unexpected_outage = st.slider("Unexpected Outage (days)", min_value=0.0, max_value=10.0, value=2.0)

# Input box to specify training runs (epochs)
training_runs = st.number_input("Number of Training Runs", min_value=100, max_value=10000, value=1000)

# Initialize variables
input_dim = 5
output_dim = 1
model_path = "models/drl_model.pth"

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Train button
if st.button("Train Model"):
    st.write(f"Training the model for {training_runs} epochs...")
    
    # Initialize and train the model
    model = DRLModel(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_train, y_train = simulate_data()
    train_model(model, optimizer, criterion, X_train, y_train, epochs=int(training_runs))

    # Save the trained model to file
    save_model(model, path=model_path)

    st.session_state.trained_model = model
    st.success(f"Model trained for {training_runs} epochs and saved to {model_path}!")

# Inference button
if st.button("Inference"):
    try:
        # Load the trained model from file if not already loaded in session state
        if st.session_state.trained_model is None:
            st.session_state.trained_model = load_model(path=model_path)

        # Prepare input data for prediction
        inputs = np.array([raw_material_1, raw_material_2, no_of_labor,
                           no_of_days_to_execute, unexpected_outage], dtype=np.float32)

        predicted_output = predict(st.session_state.trained_model, inputs)
        st.write(f"Predicted Outcome: {predicted_output:.2f} labor resources needed to execute the job on time.")
    
    except FileNotFoundError as e:
        st.error(str(e))

# What-If Analysis Section
st.subheader("What-If Analysis")
st.write("Adjust the sliders above to see how changes in parameters affect the prediction.")
