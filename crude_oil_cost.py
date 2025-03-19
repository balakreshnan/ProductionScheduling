# crude_oil_cost.py  
import streamlit as st  
import numpy as np  
import pandas as pd  
import joblib  
import os  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.compose import ColumnTransformer  
  
  
# Set the title of the Streamlit app  
st.title('Crude Oil Barrel Cost Optimization')  
  
# Define file paths for the model and preprocessor  
MODEL_PATH = 'optimized_cost_model.pth'  
PREPROCESSOR_PATH = 'preprocessor.joblib'  
DATA_PATH = 'crude_oil_dataset_200.csv'  
  
  
# 1. Data Generation  
@st.cache_data  
def generate_synthetic_data(num_samples=1000, random_state=42):  
    """  
    Generates a synthetic dataset based on the specified schema.  
    """  
    np.random.seed(random_state)  
      
    data = {  
        'crude_oil_type': np.random.choice(['Type A', 'Type B', 'Type C'], num_samples),  
        'cost_per_barrel_today': np.random.uniform(50, 100, num_samples),  
        'purchase_location': np.random.choice(['Location 1', 'Location 2', 'Location 3'], num_samples),  
        'refinery_location': np.random.choice(['Refinery 1', 'Refinery 2', 'Refinery 3'], num_samples),  
        'lead_time_days': np.random.randint(1, 30, num_samples),  
        'refinery_processing_time_days': np.random.randint(1, 10, num_samples),  
        'production_cost': np.random.uniform(20, 80, num_samples),  
        'method_of_moving': np.random.choice(['Rail', 'Ship', 'Road', 'Pipe'], num_samples),  
        'method_capacity': np.random.uniform(1000, 5000, num_samples),  
        'number_of_barrels': np.random.randint(100, 1000, num_samples),  
        'delivery_method': np.random.choice(['Ship', 'Rail', 'Pipe', 'Road'], num_samples),  
        'delivery_time_days': np.random.randint(1, 15, num_samples),  
        'delivery_location': np.random.choice(['Customer A', 'Customer B', 'Customer C'], num_samples),  
        'delivery_capacity': np.random.uniform(1000, 5000, num_samples),  
    }  
      
    df = pd.DataFrame(data)  
      
    # Create a target variable: optimized cost (a hypothetical function)  
    df['optimized_cost'] = (  
        df['cost_per_barrel_today'] +  
        df['production_cost'] +  
        (df['lead_time_days'] * 0.5) +  
        (df['refinery_processing_time_days'] * 0.3) -  
        (df['method_capacity'] * 0.01)  
    ) + np.random.normal(0, 5, num_samples)  # Adding some noise  
      
    # Save the dataset  
    df.to_csv(DATA_PATH, index=False)  
      
    return df  
  
  
# 2. Data Preprocessing  
def preprocess_data(df):  
    """  
    Preprocesses the data by encoding categorical variables and scaling numerical features.  
    """  
    # Define feature columns and target  
    X = df.drop(['optimized_cost'], axis=1)  
    y = df['optimized_cost'].values.astype(np.float32)  
      
    # Identify categorical and numerical columns  
    categorical_features = [  
        'crude_oil_type', 'purchase_location', 'refinery_location',  
        'method_of_moving', 'delivery_method', 'delivery_location'  
    ]  
    numerical_features = [  
        'cost_per_barrel_today', 'lead_time_days', 'refinery_processing_time_days',  
        'production_cost', 'method_capacity', 'number_of_barrels',  
        'delivery_time_days', 'delivery_capacity'  
    ]  
      
    # Preprocessing pipelines  
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)  
    numerical_transformer = StandardScaler()  
      
    preprocessor = ColumnTransformer(  
        transformers=[  
            ('num', numerical_transformer, numerical_features),  
            ('cat', categorical_transformer, categorical_features)  
        ]  
    )  
      
    # Fit and transform the data  
    X_processed = preprocessor.fit_transform(X)  
      
    return X_processed, y, preprocessor, X, y  
  
  
# 3. PyTorch Model Definition  
class NeuralNet(nn.Module):  
    def __init__(self, input_size):  
        super(NeuralNet, self).__init__()  
        self.network = nn.Sequential(  
            nn.Linear(input_size, 64),  
            nn.ReLU(),  
            nn.Linear(64, 32),  
            nn.ReLU(),  
            nn.Linear(32, 1)  
        )  
          
    def forward(self, x):  
        return self.network(x)  
  
  
# 4. Training Function  
def train_model_pytorch(X_train, y_train, X_val, y_val, input_size, epochs=1000, batch_size=32, learning_rate=0.001):  
    """  
    Trains the PyTorch neural network model.  
    """  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
      
    model = NeuralNet(input_size).to(device)  
      
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
      
    # Convert data to PyTorch tensors  
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  
      
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)  
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)  
      
    # Training loop  
    best_val_loss = float('inf')  
    patience = 20  
    trigger_times = 0  
      
    for epoch in range(epochs):  
        model.train()  
        optimizer.zero_grad()  
        outputs = model(X_train_tensor)  
        loss = criterion(outputs, y_train_tensor)  
        loss.backward()  
        optimizer.step()  
          
        # Validation  
        model.eval()  
        with torch.no_grad():  
            val_outputs = model(X_val_tensor)  
            val_loss = criterion(val_outputs, y_val_tensor)  
          
        # Early Stopping  
        if val_loss.item() < best_val_loss:  
            best_val_loss = val_loss.item()  
            trigger_times = 0  
            # Save the best model  
            torch.save(model.state_dict(), MODEL_PATH)  
        else:  
            trigger_times += 1  
            if trigger_times >= patience:  
                print('Early stopping!')  
                break  
          
        # Optional: Print progress  
        if (epoch+1) % 100 == 0 or epoch == 0:  
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')  
      
    # Load the best model  
    model.load_state_dict(torch.load(MODEL_PATH))  
      
    return model  
  
  
# 5. Training Procedure  
def train_model():  
    """  
    Generates data, preprocesses it, trains the model, and saves the model and preprocessor.  
    """  
    st.info("Generating synthetic data...")  
    df = generate_synthetic_data()  
      
    st.info("Preprocessing data...")  
    X_processed, y, preprocessor, X, y = preprocess_data(df)  
      
    # Split the data  
    st.info("Splitting data into training and testing sets...")  
    X_train_full, X_test, y_train_full, y_test = train_test_split(  
        X_processed, y, test_size=0.2, random_state=42  
    )  
      
    # Further split training for validation  
    X_train, X_val, y_train, y_val = train_test_split(  
        X_train_full, y_train_full, test_size=0.2, random_state=42  
    )  
      
    st.info("Building and training the PyTorch model...")  
    input_size = X_train.shape[1]  
    model = train_model_pytorch(X_train, y_train, X_val, y_val, input_size)  
      
    # Evaluate the model on test set  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model.eval()  
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)  
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)  
    with torch.no_grad():  
        predictions = model(X_test_tensor)  
        mse = nn.MSELoss()(predictions, y_test_tensor).item()  
        mae = nn.L1Loss()(predictions, y_test_tensor).item()  
      
    st.success(f'Model trained successfully! Test MAE: ${mae:.2f}')  
      
    # Save the preprocessor  
    joblib.dump(preprocessor, PREPROCESSOR_PATH)  
      
    return model, preprocessor  
  
  
# 6. Loading the Model and Preprocessor  
def load_model_and_preprocessor():  
    """  
    Loads the trained model and preprocessor from disk.  
    """  
    df = generate_synthetic_data()  # Ensure data structure for preprocessor  
    _, _, preprocessor, _, _ = preprocess_data(df)  
    preprocessor = joblib.load(PREPROCESSOR_PATH)  
      
    # Initialize the model  
    input_size = preprocessor.transform(df.drop(['optimized_cost'], axis=1)).shape[1]  
    model = NeuralNet(input_size)  
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  
    model.eval()  
      
    return model, preprocessor  
  
  
# 7. Streamlit Sidebar for Model Training  
st.sidebar.header("Model Training")  
if st.sidebar.button('Train Model'):  
    with st.spinner('Training the model. This might take a few minutes...'):  
        model, preprocessor = train_model()  
    st.sidebar.success('Model trained and saved successfully!')  
  
  
# 8. Check if model and preprocessor exist  
if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):  
    try:  
        model, preprocessor = load_model_and_preprocessor()  
        st.success("Model and preprocessor loaded successfully!")  
    except Exception as e:  
        st.error(f"Error loading model or preprocessor: {e}")  
        st.error("Please click the 'Train Model' button in the sidebar to train a new model.")  
        st.stop()  
else:  
    st.warning("No trained model found. Please train the model by clicking the 'Train Model' button in the sidebar.")  
    st.stop()  
  
  
# 9. User Input Widgets  
st.header('Input Parameters')  
  
col1, col2 = st.columns(2)  
  
with col1:  
    crude_oil_type = st.selectbox('Crude Oil Type', ['Type A', 'Type B', 'Type C'])  
    cost_per_barrel_today = st.slider('Cost per Barrel Today ($)', 50, 100, 75)  
    purchase_location = st.selectbox('Purchase Location', ['Location 1', 'Location 2', 'Location 3'])  
    refinery_location = st.selectbox('Refinery Location', ['Refinery 1', 'Refinery 2', 'Refinery 3'])  
    lead_time_days = st.slider('Lead Time to Delivery (Days)', 1, 30, 15)  
    refinery_processing_time_days = st.slider('Refinery Processing Time (Days)', 1, 10, 5)  
    production_cost = st.slider('Production Cost ($)', 20.0, 80.0, 50.0)  
  
with col2:  
    method_of_moving = st.selectbox('Method of Moving', ['Rail', 'Ship', 'Road', 'Pipe'])  
    method_capacity = st.slider('Method Capacity', 1000.0, 5000.0, 3000.0)  
    number_of_barrels = st.slider('Number of Barrels Purchased', 100, 1000, 500)  
    delivery_method = st.selectbox('Delivery Method', ['Ship', 'Rail', 'Pipe', 'Road'])  
    delivery_time_days = st.slider('Delivery Time (Days)', 1, 15, 7)  
    delivery_location = st.selectbox('Delivery Location', ['Customer A', 'Customer B', 'Customer C'])  
    delivery_capacity = st.slider('Delivery Capacity', 1000.0, 5000.0, 3000.0)  
  
  
# 10. Preparing Input Data for Prediction  
input_data = pd.DataFrame({  
    'crude_oil_type': [crude_oil_type],  
    'cost_per_barrel_today': [cost_per_barrel_today],  
    'purchase_location': [purchase_location],  
    'refinery_location': [refinery_location],  
    'lead_time_days': [lead_time_days],  
    'refinery_processing_time_days': [refinery_processing_time_days],  
    'production_cost': [production_cost],  
    'method_of_moving': [method_of_moving],  
    'method_capacity': [method_capacity],  
    'number_of_barrels': [number_of_barrels],  
    'delivery_method': [delivery_method],  
    'delivery_time_days': [delivery_time_days],  
    'delivery_location': [delivery_location],  
    'delivery_capacity': [delivery_capacity],  
})  
  
  
# 11. Prediction Button  
if st.button('Predict Optimized Cost'):  
    try:  
        # Preprocess the input data  
        X_input = preprocessor.transform(input_data)  
          
        # Convert to PyTorch tensor  
        X_input_tensor = torch.tensor(X_input, dtype=torch.float32)  
          
        # Make prediction  
        with torch.no_grad():  
            prediction = model(X_input_tensor).item()  
          
        st.success(f'**Optimized Cost: ${prediction:.2f}**')  
    except Exception as e:  
        st.error(f"Error during prediction: {e}")  
  
  
# 12. Instructions  
st.markdown("""  
---  
### Instructions:  
1. **Train the Model**: If you haven't trained the model yet, go to the sidebar and click the 'Train Model' button. This will generate synthetic data, train the model, and save it for future use.  
2. **Input Parameters**: Adjust the sliders and select options to input the parameters for the crude oil barrel you wish to analyze.  
3. **Predict**: Click the 'Predict Optimized Cost' button to see the optimized cost based on your inputs.  
""")  