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

# Simulate data for training with updated logic (including use cases)
def simulate_data(num_samples=1000):
    """
    Simulates data for training based on the updated logic:
    - More AI Engineers significantly reduce execution time.
    - UI Engineers have a smaller impact.
    - ML Engineers have half the count of AI Engineers and contribute moderately.
    - The number of use cases increases project duration linearly.
    Outputs are clipped between 7 and 150 days.
    """
    X = np.random.randint(1, 100, size=(num_samples, 9))  # Resource counts and parameters

    # Define weights for each resource based on their impact
    architects_weight = 5
    decision_makers_weight = 4
    directors_weight = 3
    managers_weight = 2
    ai_engineers_weight = -10  # Significant negative correlation (more AI engineers reduce time)
    ui_engineers_weight = -2   # Smaller negative correlation (less impact)
    ml_engineers_weight = -5   # Moderate negative correlation (half of AI engineers)
    use_cases_weight = 10      # Each use case adds time linearly

    # Compute target days based on weights and add random noise
    y = (
        X[:, 0] * architects_weight +
        X[:, 1] * decision_makers_weight +
        X[:, 2] * directors_weight +
        X[:, 3] * managers_weight +
        X[:, 4] * ai_engineers_weight +
        X[:, 5] * ui_engineers_weight +
        X[:, 6] * ml_engineers_weight +
        X[:, 7] * -15 +  # Unexpected outage has a strong negative impact on days
        X[:, 8] * use_cases_weight +  # Use cases increase project duration linearly
        np.random.randint(-10, 10, size=num_samples)  # Add random noise
    ).clip(14, 150).astype(np.float32)  # Clip values between 7 and 150 days

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
def save_model(model, path="models/aiprj_model.pth"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if it doesn't exist
    torch.save(model.state_dict(), path)

# Load the trained model from a file
def load_model(path="models/aiprj_model.pth", input_dim=9, output_dim=1):
    model = DRLModel(input_dim=input_dim, output_dim=output_dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation mode
    else:
        raise FileNotFoundError(f"Model file not found at {path}")
    return model

# Streamlit UI
st.title("Deep Reinforcement Learning: Project Execution Time Prediction")

# User input sliders for resources and project parameters
architects = st.slider("Architects", min_value=1, max_value=10, value=1)
decision_makers = st.slider("Decision Makers", min_value=0, max_value=1, value=1)
directors = st.slider("Directors", min_value=0, max_value=2, value=0)
managers = st.slider("Managers", min_value=1, max_value=3, value=1)

ai_engineers = st.slider("AI Engineers", min_value=1, max_value=50, value=2)
ui_engineers = st.slider("UI Engineers (for web development)", min_value=1, max_value=20, value=1)
ml_engineers = st.slider("ML Engineers (half of AI Engineers)", min_value=1, max_value=25,
                         value=max(1, ai_engineers // 2))

unexpected_outage = st.slider("Unexpected Outage (days)", min_value=0.0, max_value=10.0, value=2.0)

# Input box for specifying number of use cases
use_cases = st.number_input("Number of Use Cases", min_value=1, max_value=1000, value=4)

# Input box for specifying training runs (epochs)
training_runs = st.number_input("Number of Training Iterations", min_value=100, max_value=20000, value=1000)

# Initialize variables
input_dim = 9   # Updated input dimension to include "use_cases"
output_dim = 1
model_path = "models/aiprj_model.pth"

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

# Train button
if st.button("Train Model"):
    st.write(f"Training the model for {training_runs} iterations...")
    
    # Initialize and train the model
    model = DRLModel(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    X_train, y_train = simulate_data()
    train_model(model, optimizer, criterion, X_train, y_train, epochs=int(training_runs))

    # Save the trained model to file
    save_model(model, path=model_path)

    st.session_state.trained_model = model
    st.success(f"Model trained for {training_runs} iterations and saved to {model_path}!")

# Inference button
if st.button("Inference"):
    try:
        # Load the trained model from file if not already loaded in session state
        if st.session_state.trained_model is None:
            st.session_state.trained_model = load_model(path=model_path)

        # Prepare input data for prediction
        inputs = np.array([architects,
                           decision_makers,
                           directors,
                           managers,
                           ai_engineers,
                           ui_engineers,
                           ml_engineers,
                           unexpected_outage,
                           use_cases], dtype=np.float32)

        predicted_output = predict(st.session_state.trained_model, inputs)

        # Clip prediction to ensure it stays within range [7-150]
        predicted_output_clipped = np.clip(predicted_output, 14.0, 150.0)

        st.write(f"Predicted Outcome: The project will take approximately {predicted_output_clipped:.2f} days to complete.")
    
    except FileNotFoundError as e:
        st.error(str(e))

# What-If Analysis Section
st.subheader("What-If Analysis")
st.write("Adjust the sliders above to see how changes in resources and parameters affect the prediction.")
