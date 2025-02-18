import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial import KDTree
import openpyxl
from tqdm import tqdm
import math
import random
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Environment setup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ST_TGN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout=0.5):
        super(ST_TGN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_features, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, 
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, edge_attr):
        batch_size, window_size, num_stations, num_features = x.shape
        x = x.view(-1, num_features)
        
        for conv in self.gcn_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        x = x.view(batch_size, window_size, num_stations, -1)
        x = x.mean(dim=2)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

class ClimateDataset(Dataset):
    def __init__(self, df, input_features, target, window_size=24, step_size=6, num_stations=1):
        super(ClimateDataset, self).__init__()
        self.target = target
        self.window_size = window_size
        self.step_size = step_size
        self.df = df
        self.input_features = input_features
        self.num_stations = num_stations
        self.windows = self.create_sliding_windows()
    
    def create_sliding_windows(self):
        data_list = []
        grouped = self.df.groupby('station_id')
        
        for station_id, group in grouped:
            group = group.sort_values('DATETIME').reset_index(drop=True)
            total_time_steps = len(group)
            
            for start in range(0, total_time_steps - self.window_size, self.step_size):
                end = start + self.window_size
                window = group.iloc[start:end]
                
                if len(window) != self.window_size:
                    continue
                
                x = window[self.input_features].values
                x = x.reshape(self.window_size, 1, -1)
                x = np.repeat(x, self.num_stations, axis=1)
                x = torch.tensor(x, dtype=torch.float)
                
                y = window[self.target].iloc[-1]
                y_tensor = torch.tensor([y], dtype=torch.float)
                
                data_list.append((x, y_tensor))
        
        return data_list
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return x, y

def main():
    # Set random seed
    set_seed(42)
    
    # Load data
    df = pd.read_csv('sample_data.csv')
    df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR']])
    
    # Create station IDs
    stations = df[['LATITUDE', 'LONGITUDE']].drop_duplicates().reset_index(drop=True)
    stations['station_id'] = stations.index
    df = df.merge(stations, on=['LATITUDE', 'LONGITUDE'], how='left')
    
    # Split stations using K-Means
    coordinates = stations[['LATITUDE', 'LONGITUDE']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(coordinates)
    stations['cluster'] = clusters
    
    # Split into train/val/test
    train_stations = []
    val_stations = []
    test_stations = []
    
    for cluster in range(kmeans.n_clusters):
        cluster_stations = stations[stations['cluster'] == cluster]
        n_stations = len(cluster_stations)
        train_size = int(0.7 * n_stations)
        val_size = int(0.15 * n_stations)
        
        indices = np.random.permutation(n_stations)
        train_stations.extend(cluster_stations.iloc[indices[:train_size]]['station_id'].tolist())
        val_stations.extend(cluster_stations.iloc[indices[train_size:train_size+val_size]]['station_id'].tolist())
        test_stations.extend(cluster_stations.iloc[indices[train_size+val_size:]]['station_id'].tolist())
    
    # Build graph
    coordinates = stations[['LATITUDE', 'LONGITUDE']].values
    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(coordinates, k=5)
    
    edge_index = []
    edge_attr = []
    
    for i in range(len(stations)):
        for j in range(1, 5):  # Skip self (j=0)
            neighbor = indices[i][j]
            distance = distances[i][j]
            edge_index.append([i, neighbor])
            edge_attr.append(distance)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
    
    # Prepare features and target
    features = ['Temperature_K', 'Specific_Humidity_gkg']
    target = 'Wind_Speed_ms'
    
    # Create datasets
    train_df = df[df['station_id'].isin(train_stations)].copy()
    val_df = df[df['station_id'].isin(val_stations)].copy()
    test_df = df[df['station_id'].isin(test_stations)].copy()
    
    # Scale features
    scaler = StandardScaler()
    train_df[features + [target]] = scaler.fit_transform(train_df[features + [target]])
    val_df[features + [target]] = scaler.transform(val_df[features + [target]])
    test_df[features + [target]] = scaler.transform(test_df[features + [target]])
    
    # Create data loaders
    train_dataset = ClimateDataset(train_df, features, target, num_stations=len(stations))
    val_dataset = ClimateDataset(val_df, features, target, num_stations=len(stations))
    test_dataset = ClimateDataset(test_df, features, target, num_stations=len(stations))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ST_TGN(
        num_features=len(features),
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training'):
            optimizer.zero_grad()
            x = batch_x.to(device)
            y = batch_y.to(device)
            output = model(x, edge_index, edge_attr)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc='Validation'):
                x = batch_x.to(device)
                y = batch_y.to(device)
                output = model(x, edge_index, edge_attr)
                val_loss += criterion(output, y).item()
                val_preds.extend(output.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_bias = np.mean(val_preds - val_true)
        val_mae = mean_absolute_error(val_true, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Metrics - Bias: {val_bias:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Test evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing'):
            x = batch_x.to(device)
            y = batch_y.to(device)
            output = model(x, edge_index, edge_attr)
            test_preds.extend(output.cpu().numpy())
            test_true.extend(y.cpu().numpy())
    
    test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))
    test_true = scaler.inverse_transform(np.array(test_true).reshape(-1, 1))
    
    test_bias = np.mean(test_preds - test_true)
    test_mae = mean_absolute_error(test_true, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
    
    print('\nTest Results:')
    print(f'  Bias: {test_bias:.4f}')
    print(f'  MAE: {test_mae:.4f}')
    print(f'  RMSE: {test_rmse:.4f}')
    
    # Save results
    results = pd.DataFrame({
        'True_Values': test_true.flatten(),
        'Predicted_Values': test_preds.flatten()
    })
    results.to_csv('test_results.csv', index=False)
    print("\nResults saved to test_results.csv")

if __name__ == '__main__':
    main()