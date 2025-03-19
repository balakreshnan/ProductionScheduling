import pandas as pd  
import numpy as np  
  
# Set random seed for reproducibility  
np.random.seed(42)  
  
# Define possible categories for categorical variables  
crude_types = ['Brent', 'WTI', 'Dubai', 'OPEC Basket', 'Bonny Light']  
purchase_locations = ['Houston', 'New York', 'London', 'Singapore', 'Dubai', 'Rotterdam']  
refinery_locations = ['Houston Refinery', 'New York Refinery', 'Singapore Refinery', 'Dubai Refinery', 'Rotterdam Refinery']  
methods_of_moving = ['Rail', 'Ship', 'Road', 'Pipe']  
delivery_methods = ['Ship', 'Rail', 'Pipe', 'Road']  
customer_locations = ['Los Angeles', 'New York', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Boston', 'Atlanta', 'Denver', 'San Francisco']  
  
# Generate 200 rows of data  
num_rows = 200  
  
# Function to generate capacity based on method  
def generate_capacity(method, capacity_type='Method'):  
    if method == 'Pipe':  
        return np.random.randint(10000, 50000)  
    elif method == 'Rail':  
        return np.random.randint(1000, 10000)  
    elif method == 'Ship':  
        return np.random.randint(50000, 200000)  
    else:  # Road  
        return np.random.randint(500, 5000)  
  
# Generate data  
data = {  
    'Crude Oil Selection': np.random.choice(crude_types, num_rows),  
    'Cost of Barrel Today ($)': np.round(np.random.uniform(50, 100, num_rows), 2),  
    'Purchase Location': np.random.choice(purchase_locations, num_rows),  
    'Refinery Location': np.random.choice(refinery_locations, num_rows),  
    'Lead Time to Delivery (days)': np.random.randint(1, 15, num_rows),  
    'Refinery Processing Time (days)': np.random.randint(1, 10, num_rows),  
    'Cost of Production ($)': np.round(np.random.uniform(30, 80, num_rows), 2),  
    'Method of Moving': np.random.choice(methods_of_moving, num_rows),  
    'Method Capacity': [generate_capacity(method) for method in np.random.choice(methods_of_moving, num_rows)],  
    'Number of Barrels': np.random.randint(100, 10000, num_rows),  
    'Cost per Barrel ($)': np.round(np.random.uniform(50, 100, num_rows), 2),  
    'Delivery Method': np.random.choice(delivery_methods, num_rows),  
    'Delivery Time (days)': np.random.randint(1, 20, num_rows),  
    'Delivery Location': np.random.choice(customer_locations, num_rows),  
    'Delivery Capacity': [generate_capacity(method, 'Delivery') for method in np.random.choice(delivery_methods, num_rows)]  
}  
  
df = pd.DataFrame(data)  
  
# Save the DataFrame to a CSV file  
df.to_csv('crude_oil_dataset_200.csv', index=False)  
  
print("Dataset 'crude_oil_dataset_200.csv' with 200 rows has been generated successfully.")  