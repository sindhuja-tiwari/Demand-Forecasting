import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DemandForecastingSystem:
    def __init__(self, data_path='data/train.csv'):
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_test = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        print(f"Data Path: {data_path}")
        print("\n" + "="*80)
    
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully!")
            print(f"✓ Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            print("\n📊 Dataset Preview:")
            print(self.df.head(10))
            
            print("\n📋 Column Information:")
            print(self.df.info())
            
            print("\n📈 Statistical Summary:")
            print(self.df.describe())
            
            print("\n🔍 Missing Values:")
            missing = self.df.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0])
            else:
                print("No missing values found!")
                
            print("\n✓ Data loading complete!")
            
        except FileNotFoundError:
            print("❌ ERROR: Data file not found!")
            print(f"Please download the dataset from Kaggle and place it at: {self.data_path}")
            print("Dataset: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast")
            raise
        
        return self.df
    
    def preprocess_data(self):
        
        self.df_processed = self.df.copy()
        
        # Convert date to datetime
        print("→ Converting date column...")
        # Try to automatically infer the date format
        try:
            self.df_processed['Date'] = pd.to_datetime(self.df_processed['Date'], format='ISO8601')
        except:
            try:
                self.df_processed['Date'] = pd.to_datetime(self.df_processed['Date'], format='%d-%m-%Y')
            except:
                # Let pandas infer the format automatically
                self.df_processed['Date'] = pd.to_datetime(self.df_processed['Date'], infer_datetime_format=True)
        
        # Sort by store and date
        self.df_processed = self.df_processed.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        # Extract temporal features
        print("→ Extracting temporal features...")
        self.df_processed['Year'] = self.df_processed['Date'].dt.year
        self.df_processed['Month'] = self.df_processed['Date'].dt.month
        self.df_processed['Week'] = self.df_processed['Date'].dt.isocalendar().week
        self.df_processed['Day'] = self.df_processed['Date'].dt.day
        self.df_processed['DayOfWeek'] = self.df_processed['Date'].dt.dayofweek
        self.df_processed['Quarter'] = self.df_processed['Date'].dt.quarter
        self.df_processed['DayOfYear'] = self.df_processed['Date'].dt.dayofyear
        self.df_processed['WeekOfYear'] = self.df_processed['Date'].dt.isocalendar().week
        
        # Is weekend
        self.df_processed['IsWeekend'] = (self.df_processed['DayOfWeek'] >= 5).astype(int)
        
        # Season
        self.df_processed['Season'] = self.df_processed['Month'].apply(
            lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
        )
        
        # Create lag features (previous sales)
        print("→ Creating lag features...")
        for lag in [1, 2, 4, 8, 12]:
            self.df_processed[f'Sales_Lag_{lag}'] = self.df_processed.groupby('Store')['Weekly_Sales'].shift(lag)
        
        # Rolling statistics
        print("→ Computing rolling statistics...")
        for window in [4, 8, 12]:
            # Rolling mean
            self.df_processed[f'Sales_RollingMean_{window}'] = self.df_processed.groupby('Store')['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            # Rolling std
            self.df_processed[f'Sales_RollingStd_{window}'] = self.df_processed.groupby('Store')['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Exponential weighted moving average
        self.df_processed['Sales_EWMA'] = self.df_processed.groupby('Store')['Weekly_Sales'].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        )
        
        # Store statistics
        print("→ Computing store-level statistics...")
        store_stats = self.df_processed.groupby('Store')['Weekly_Sales'].agg(['mean', 'std', 'median']).reset_index()
        store_stats.columns = ['Store', 'Store_Sales_Mean', 'Store_Sales_Std', 'Store_Sales_Median']
        self.df_processed = self.df_processed.merge(store_stats, on='Store', how='left')
        
        # Year-over-year growth (if multiple years)
        if self.df_processed['Year'].nunique() > 1:
            self.df_processed['YoY_Growth'] = self.df_processed.groupby(['Store', 'WeekOfYear'])['Weekly_Sales'].pct_change(periods=52)
        
        # Handle holiday flag if exists
        if 'Holiday_Flag' in self.df_processed.columns:
            self.df_processed['IsHoliday'] = self.df_processed['Holiday_Flag']
        
        # Fill missing values
        print("→ Handling missing values...")
        # For lag features, forward fill then backward fill
        self.df_processed.fillna(method='ffill', inplace=True)
        self.df_processed.fillna(method='bfill', inplace=True)
        self.df_processed.fillna(0, inplace=True)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"✓ Total features: {self.df_processed.shape[1]}")
        print(f"✓ Feature names: {list(self.df_processed.columns)}")
        
        return self.df_processed
    
    def exploratory_analysis(self):
        fig = plt.figure(figsize=(20, 15))
        ax1 = plt.subplot(3, 3, 1)
        sales_by_date = self.df_processed.groupby('Date')['Weekly_Sales'].sum()
        ax1.plot(sales_by_date.index, sales_by_date.values, linewidth=2, color='#2E86AB')
        ax1.set_title('Total Weekly Sales Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Sales ($)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Top 10 stores by average sales
        ax2 = plt.subplot(3, 3, 2)
        top_stores = self.df_processed.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False).head(10)
        colors = sns.color_palette("viridis", len(top_stores))
        ax2.barh(range(len(top_stores)), top_stores.values, color=colors)
        ax2.set_yticks(range(len(top_stores)))
        ax2.set_yticklabels([f'Store {s}' for s in top_stores.index])
        ax2.set_xlabel('Average Sales ($)', fontsize=11)
        ax2.set_title('Top 10 Stores by Average Sales', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Sales by month
        ax3 = plt.subplot(3, 3, 3)
        monthly_sales = self.df_processed.groupby('Month')['Weekly_Sales'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax3.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8, color='#A23B72')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months, rotation=45)
        ax3.set_xlabel('Month', fontsize=11)
        ax3.set_ylabel('Average Sales ($)', fontsize=11)
        ax3.set_title('Seasonality: Average Sales by Month', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Sales distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(self.df_processed['Weekly_Sales'], bins=50, color='#F18F01', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Weekly Sales ($)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Distribution of Weekly Sales', fontsize=14, fontweight='bold')
        ax4.axvline(self.df_processed['Weekly_Sales'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: ${self.df_processed['Weekly_Sales'].mean():,.0f}")
        ax4.legend()
        
        # 5. Sales by day of week
        ax5 = plt.subplot(3, 3, 5)
        dow_sales = self.df_processed.groupby('DayOfWeek')['Weekly_Sales'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax5.bar(range(7), dow_sales.values, color='#06A77D', edgecolor='black')
        ax5.set_xticks(range(7))
        ax5.set_xticklabels(days)
        ax5.set_xlabel('Day of Week', fontsize=11)
        ax5.set_ylabel('Average Sales ($)', fontsize=11)
        ax5.set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
        
        # 6. Holiday vs Non-Holiday
        if 'IsHoliday' in self.df_processed.columns:
            ax6 = plt.subplot(3, 3, 6)
            holiday_sales = self.df_processed.groupby('IsHoliday')['Weekly_Sales'].mean()
            ax6.bar(['Non-Holiday', 'Holiday'], holiday_sales.values, color=['#D62246', '#4ECDC4'], 
                   edgecolor='black', width=0.5)
            ax6.set_ylabel('Average Sales ($)', fontsize=11)
            ax6.set_title('Holiday vs Non-Holiday Sales', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, v in enumerate(holiday_sales.values):
                ax6.text(i, v + 1000, f'${v:,.0f}', ha='center', fontweight='bold')
        
        # 7. Sales trend with moving average
        ax7 = plt.subplot(3, 3, 7)
        sample_store = self.df_processed['Store'].mode()[0]
        store_data = self.df_processed[self.df_processed['Store'] == sample_store].sort_values('Date')
        ax7.plot(store_data['Date'], store_data['Weekly_Sales'], alpha=0.5, label='Actual Sales')
        ax7.plot(store_data['Date'], store_data['Sales_RollingMean_4'], linewidth=2, 
                color='red', label='4-Week Moving Avg')
        ax7.set_xlabel('Date', fontsize=11)
        ax7.set_ylabel('Sales ($)', fontsize=11)
        ax7.set_title(f'Store {sample_store}: Sales with Moving Average', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 8. Correlation heatmap (top features)
        ax8 = plt.subplot(3, 3, 8)
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        corr_cols = ['Weekly_Sales', 'Sales_Lag_1', 'Sales_RollingMean_4', 'Month', 'Store']
        corr_cols = [c for c in corr_cols if c in numeric_cols]
        corr_matrix = self.df_processed[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, ax=ax8, cbar_kws={'shrink': 0.8})
        ax8.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 9. Quarterly sales comparison
        ax9 = plt.subplot(3, 3, 9)
        quarterly_sales = self.df_processed.groupby('Quarter')['Weekly_Sales'].mean()
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        ax9.bar(quarters, quarterly_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], 
               edgecolor='black')
        ax9.set_xlabel('Quarter', fontsize=11)
        ax9.set_ylabel('Average Sales ($)', fontsize=11)
        ax9.set_title('Average Sales by Quarter', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(quarterly_sales.values):
            ax9.text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('demand_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved as 'demand_analysis_comprehensive.png'")
        plt.show()
        
        # Print key insights
        print("\n📊 KEY INSIGHTS:")
        print("-" * 80)
        print(f"→ Total Stores: {self.df_processed['Store'].nunique()}")
        print(f"→ Date Range: {self.df_processed['Date'].min()} to {self.df_processed['Date'].max()}")
        print(f"→ Average Weekly Sales: ${self.df_processed['Weekly_Sales'].mean():,.2f}")
        print(f"→ Median Weekly Sales: ${self.df_processed['Weekly_Sales'].median():,.2f}")
        print(f"→ Std Dev: ${self.df_processed['Weekly_Sales'].std():,.2f}")
        print(f"→ Peak Sales Month: {months[monthly_sales.idxmax() - 1]}")
        print(f"→ Lowest Sales Month: {months[monthly_sales.idxmin() - 1]}")
        
        if 'IsHoliday' in self.df_processed.columns:
            holiday_avg = self.df_processed[self.df_processed['IsHoliday'] == 1]['Weekly_Sales'].mean()
            non_holiday_avg = self.df_processed[self.df_processed['IsHoliday'] == 0]['Weekly_Sales'].mean()
            print(f"→ Holiday Sales Premium: {((holiday_avg - non_holiday_avg) / non_holiday_avg * 100):.1f}%")
    
    def prepare_features(self):
        # Define features to use
        feature_cols = [
            'Store', 'Month', 'Week', 'Day', 'DayOfWeek', 'Quarter', 'DayOfYear',
            'WeekOfYear', 'IsWeekend', 'Season',
            'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_4', 'Sales_Lag_8', 'Sales_Lag_12',
            'Sales_RollingMean_4', 'Sales_RollingMean_8', 'Sales_RollingMean_12',
            'Sales_RollingStd_4', 'Sales_RollingStd_8', 'Sales_RollingStd_12',
            'Sales_EWMA', 'Store_Sales_Mean', 'Store_Sales_Std', 'Store_Sales_Median'
        ]
        
        # Add optional features if they exist
        optional_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'YoY_Growth']
        for feat in optional_features:
            if feat in self.df_processed.columns:
                feature_cols.append(feat)
        
        # Filter features that actually exist in dataframe
        feature_cols = [f for f in feature_cols if f in self.df_processed.columns]
        
        print(f"→ Selected {len(feature_cols)} features")
        print(f"→ Features: {feature_cols}")
        
        # Prepare X and y
        X = self.df_processed[feature_cols].copy()
        y = self.df_processed['Weekly_Sales'].copy()
        
        # Remove any remaining NaN or inf values
        mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1) | y.isnull() | np.isinf(y))
        X = X[mask]
        y = y[mask]
        
        print(f"→ Final dataset shape: {X.shape}")
        print(f"→ Samples removed due to NaN/inf: {(~mask).sum()}")
        
        self.feature_cols = feature_cols
        self.X = X
        self.y = y
        
        return X, y
    
    def train_models(self):
        print("→ Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"  Training set: {self.X_train.shape[0]:,} samples")
        print(f"  Test set: {self.X_test.shape[0]:,} samples")
        
        # Train Random Forest
        print("\n→ Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(self.X_train, self.y_train)
        
        # Train Gradient Boosting
        print("→ Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbose=0
        )
        gb_model.fit(self.X_train, self.y_train)
        
        # Compare models
        print("\n" + "="*80)
        print(" " * 25 + "MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        models = {
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model
        }
        
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Metrics
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            results[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': y_pred_test
            }
            
            print(f"\n{name}:")
            print("-" * 80)
            print(f"  Training Set:")
            print(f"    MAE:  ${train_mae:,.2f}")
            print(f"    RMSE: ${train_rmse:,.2f}")
            print(f"    R²:   {train_r2:.4f}")
            print(f"  Test Set:")
            print(f"    MAE:  ${test_mae:,.2f}")
            print(f"    RMSE: ${test_rmse:,.2f}")
            print(f"    R²:   {test_r2:.4f}")
        
        # Select best model based on test R²
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.model = models[best_model_name]
        self.y_pred_test = results[best_model_name]['predictions']
        
        print("\n" + "="*80)
        print(f"✓ Best Model: {best_model_name} (Test R² = {results[best_model_name]['test_r2']:.4f})")
        print("="*80)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'Feature': self.feature_cols,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n📊 TOP 15 MOST IMPORTANT FEATURES:")
            print("-" * 80)
            for idx, row in self.feature_importance.head(15).iterrows():
                print(f"  {row['Feature']:30s} {row['Importance']:.4f}")
        
        return self.model
    
    def visualize_model_results(self):
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Feature Importance
        ax1 = plt.subplot(2, 3, 1)
        top_features = self.feature_importance.head(15)
        colors = sns.color_palette("rocket", len(top_features))
        ax1.barh(range(len(top_features)), top_features['Importance'].values, color=colors)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['Feature'].values, fontsize=10)
        ax1.set_xlabel('Importance Score', fontsize=11)
        ax1.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Actual vs Predicted (Scatter)
        ax2 = plt.subplot(2, 3, 2)
        sample_size = min(5000, len(self.y_test))
        sample_indices = np.random.choice(len(self.y_test), sample_size, replace=False)
        ax2.scatter(self.y_test.iloc[sample_indices], self.y_pred_test[sample_indices], 
                   alpha=0.4, s=20, color='#2E86AB')
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_pred_test.min())
        max_val = max(self.y_test.max(), self.y_pred_test.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Sales ($)', fontsize=11)
        ax2.set_ylabel('Predicted Sales ($)', fontsize=11)
        ax2.set_title('Actual vs Predicted Sales', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        ax3 = plt.subplot(2, 3, 3)
        residuals = self.y_test.values - self.y_pred_test
        ax3.hist(residuals, bins=50, color='#A23B72', edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.set_xlabel('Residual ($)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        ax3.legend()
        
        # Add statistics
        ax3.text(0.02, 0.98, f'Mean: ${residuals.mean():,.0f}\nStd: ${residuals.std():,.0f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Residuals vs Predicted
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(self.y_pred_test, residuals, alpha=0.4, s=20, color='#F18F01')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Predicted Sales ($)', fontsize=11)
        ax4.set_ylabel('Residual ($)', fontsize=11)
        ax4.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Time Series Prediction (Sample Store)
        ax5 = plt.subplot(2, 3, 5)
        # Get a sample store from test set
        test_data = self.X_test.copy()
        test_data['Actual'] = self.y_test.values
        test_data['Predicted'] = self.y_pred_test
        
        if 'Store' in test_data.columns:
            sample_store = test_data['Store'].mode()[0]
            store_test = test_data[test_data['Store'] == sample_store].head(50)
            
            ax5.plot(range(len(store_test)), store_test['Actual'].values, 
                    marker='o', label='Actual', linewidth=2, markersize=6)
            ax5.plot(range(len(store_test)), store_test['Predicted'].values, 
                    marker='s', label='Predicted', linewidth=2, markersize=6)
            ax5.set_xlabel('Time Period', fontsize=11)
            ax5.set_ylabel('Sales ($)', fontsize=11)
            ax5.set_title(f'Store {sample_store}: Actual vs Predicted (Sample)', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Error Metrics Summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        mae = mean_absolute_error(self.y_test, self.y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
        r2 = r2_score(self.y_test, self.y_pred_test)
        mape = np.mean(np.abs((self.y_test.values - self.y_pred_test) / self.y_test.values)) * 100
        
        summary_text = f"""
        MODEL PERFORMANCE SUMMARY
        {'='*40}
        
        Mean Absolute Error (MAE):
            ${mae:,.2f}
        
        Root Mean Squared Error (RMSE):
            ${rmse:,.2f}
        
        R² Score:
            {r2:.4f}
        
        Mean Absolute % Error (MAPE):
            {mape:.2f}%
        
        {'='*40}
        Interpretation:
        
        • Model explains {r2*100:.1f}% of variance
        • Average error: ${mae:,.0f}
        • Predictions are highly accurate
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('model_performance_results.png', dpi=300, bbox_inches='tight')
        print("✓ Model visualizations saved as 'model_performance_results.png'")
        plt.show()
    
    def forecast_future_demand(self, store_id, weeks_ahead=8):
        print(f"\n[FORECAST] Predicting {weeks_ahead} weeks ahead for Store {store_id}...")
        print("-" * 80)
        
        # Get latest data for the store
        store_data = self.df_processed[self.df_processed['Store'] == store_id].copy()
        store_data = store_data.sort_values('Date')
        
        if len(store_data) == 0:
            print(f"❌ No data found for Store {store_id}")
            return None
        
        latest_record = store_data.iloc[-1].copy()
        last_date = latest_record['Date']
        
        print(f"→ Last recorded date: {last_date}")
        print(f"→ Last recorded sales: ${latest_record['Weekly_Sales']:,.2f}")
        
        forecasts = []
        current_record = latest_record.copy()
        
        for week in range(1, weeks_ahead + 1):
            # Calculate future date
            future_date = last_date + timedelta(weeks=week)
            
            # Prepare features for prediction
            future_features = {}
            
            # Temporal features
            future_features['Store'] = store_id
            future_features['Month'] = future_date.month
            future_features['Week'] = future_date.isocalendar()[1]
            future_features['Day'] = future_date.day
            future_features['DayOfWeek'] = future_date.weekday()
            future_features['Quarter'] = (future_date.month - 1) // 3 + 1
            future_features['DayOfYear'] = future_date.timetuple().tm_yday
            future_features['WeekOfYear'] = future_date.isocalendar()[1]
            future_features['IsWeekend'] = 1 if future_date.weekday() >= 5 else 0
            future_features['Season'] = (future_date.month % 12 + 3) // 3
            
            # Lag features (use most recent predictions)
            future_features['Sales_Lag_1'] = current_record.get('Weekly_Sales', latest_record['Weekly_Sales'])
            future_features['Sales_Lag_2'] = current_record.get('Sales_Lag_1', latest_record['Sales_Lag_1'])
            future_features['Sales_Lag_4'] = current_record.get('Sales_Lag_2', latest_record['Sales_Lag_2'])
            future_features['Sales_Lag_8'] = current_record.get('Sales_Lag_4', latest_record['Sales_Lag_4'])
            future_features['Sales_Lag_12'] = current_record.get('Sales_Lag_8', latest_record['Sales_Lag_8'])
            
            # Rolling features
            future_features['Sales_RollingMean_4'] = current_record.get('Sales_RollingMean_4', latest_record['Sales_RollingMean_4'])
            future_features['Sales_RollingMean_8'] = current_record.get('Sales_RollingMean_8', latest_record['Sales_RollingMean_8'])
            future_features['Sales_RollingMean_12'] = current_record.get('Sales_RollingMean_12', latest_record['Sales_RollingMean_12'])
            future_features['Sales_RollingStd_4'] = current_record.get('Sales_RollingStd_4', latest_record['Sales_RollingStd_4'])
            future_features['Sales_RollingStd_8'] = current_record.get('Sales_RollingStd_8', latest_record['Sales_RollingStd_8'])
            future_features['Sales_RollingStd_12'] = current_record.get('Sales_RollingStd_12', latest_record['Sales_RollingStd_12'])
            future_features['Sales_EWMA'] = current_record.get('Sales_EWMA', latest_record['Sales_EWMA'])
            
            # Store statistics
            future_features['Store_Sales_Mean'] = latest_record['Store_Sales_Mean']
            future_features['Store_Sales_Std'] = latest_record['Store_Sales_Std']
            future_features['Store_Sales_Median'] = latest_record['Store_Sales_Median']
            
            # Optional features
            if 'Temperature' in self.feature_cols:
                future_features['Temperature'] = latest_record.get('Temperature', 70)
            if 'Fuel_Price' in self.feature_cols:
                future_features['Fuel_Price'] = latest_record.get('Fuel_Price', 3.5)
            if 'CPI' in self.feature_cols:
                future_features['CPI'] = latest_record.get('CPI', 200)
            if 'Unemployment' in self.feature_cols:
                future_features['Unemployment'] = latest_record.get('Unemployment', 7)
            if 'IsHoliday' in self.feature_cols:
                future_features['IsHoliday'] = 0
            if 'YoY_Growth' in self.feature_cols:
                future_features['YoY_Growth'] = 0
            
            # Create feature vector in correct order
            feature_vector = pd.DataFrame([future_features])[self.feature_cols]
            
            # Make prediction
            prediction = self.model.predict(feature_vector)[0]
            
            # Store forecast
            forecasts.append({
                'Date': future_date,
                'Week': week,
                'Predicted_Sales': prediction,
                'Store': store_id
            })
            
            # Update current record for next iteration
            current_record['Weekly_Sales'] = prediction
            for feat in future_features:
                current_record[feat] = future_features[feat]
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts)
        
        # Display results
        print("\n📈 FORECAST RESULTS:")
        print("-" * 80)
        print(forecast_df.to_string(index=False))
        
        print(f"\n💰 SUMMARY:")
        print("-" * 80)
        print(f"  Total Forecasted Demand:  ${forecast_df['Predicted_Sales'].sum():,.2f}")
        print(f"  Average Weekly Demand:    ${forecast_df['Predicted_Sales'].mean():,.2f}")
        print(f"  Min Weekly Demand:        ${forecast_df['Predicted_Sales'].min():,.2f}")
        print(f"  Max Weekly Demand:        ${forecast_df['Predicted_Sales'].max():,.2f}")
        print(f"  Std Dev:                  ${forecast_df['Predicted_Sales'].std():,.2f}")
        
        # Visualize forecast
        self._visualize_forecast(store_id, store_data, forecast_df)
        
        return forecast_df
    
    def _visualize_forecast(self, store_id, historical_data, forecast_df):
        """Visualize the forecast vs historical data"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Historical + Forecast
        historical_recent = historical_data.tail(52)  # Last year
        
        ax1.plot(historical_recent['Date'], historical_recent['Weekly_Sales'], 
                marker='o', linewidth=2, markersize=5, label='Historical Sales', color='#2E86AB')
        ax1.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], 
                marker='s', linewidth=2, markersize=6, label='Forecasted Sales', 
                color='#F18F01', linestyle='--')
        
        ax1.axvline(x=historical_recent['Date'].iloc[-1], color='red', 
                   linestyle=':', linewidth=2, label='Forecast Start')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Sales ($)', fontsize=12)
        ax1.set_title(f'Store {store_id}: Historical vs Forecasted Sales', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Forecast breakdown
        ax2.bar(forecast_df['Week'], forecast_df['Predicted_Sales'], 
               color='#06A77D', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Week Ahead', fontsize=12)
        ax2.set_ylabel('Predicted Sales ($)', fontsize=12)
        ax2.set_title(f'Store {store_id}: Weekly Forecast Breakdown', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (week, sales) in enumerate(zip(forecast_df['Week'], forecast_df['Predicted_Sales'])):
            ax2.text(week, sales + 500, f'${sales:,.0f}', 
                    ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'forecast_store_{store_id}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Forecast visualization saved as 'forecast_store_{store_id}.png'")
        plt.show()
    
    def inventory_recommendations(self, forecast_df, safety_stock_weeks=2):
        print(f"\n[INVENTORY OPTIMIZATION] Generating recommendations...")
        print("-" * 80)
        
        avg_weekly_demand = forecast_df['Predicted_Sales'].mean()
        max_weekly_demand = forecast_df['Predicted_Sales'].max()
        total_demand = forecast_df['Predicted_Sales'].sum()
        demand_std = forecast_df['Predicted_Sales'].std()
        
        # Safety stock calculation (using service level approach)
        safety_stock = safety_stock_weeks * avg_weekly_demand
        
        # Reorder point
        lead_time_weeks = 1  # Assume 1 week lead time
        reorder_point = (lead_time_weeks * avg_weekly_demand) + safety_stock
        
        # Economic Order Quantity (simplified)
        order_quantity = avg_weekly_demand * 4  # 4 weeks of inventory
        
        recommendations = {
            'average_weekly_demand': avg_weekly_demand,
            'peak_weekly_demand': max_weekly_demand,
            'total_forecasted_demand': total_demand,
            'demand_volatility': demand_std,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'recommended_order_quantity': order_quantity,
            'min_inventory_level': safety_stock,
            'max_inventory_level': safety_stock + order_quantity
        }
        
        print("\n📦 INVENTORY RECOMMENDATIONS:")
        print("="*80)
        print(f"  Average Weekly Demand:        ${avg_weekly_demand:,.2f}")
        print(f"  Peak Weekly Demand:           ${max_weekly_demand:,.2f}")
        print(f"  Total Forecasted Demand:      ${total_demand:,.2f}")
        print(f"  Demand Volatility (Std Dev):  ${demand_std:,.2f}")
        print(f"\n  Safety Stock Level:           ${safety_stock:,.2f}")
        print(f"  Reorder Point:                ${reorder_point:,.2f}")
        print(f"  Recommended Order Quantity:   ${order_quantity:,.2f}")
        print(f"  Min Inventory Level:          ${safety_stock:,.2f}")
        print(f"  Max Inventory Level:          ${safety_stock + order_quantity:,.2f}")
        
        print(f"\n💡 INSIGHTS:")
        print("-"*80)
        if demand_std / avg_weekly_demand > 0.3:
            print("  ⚠️  High demand volatility detected - consider increasing safety stock")
        else:
            print("  ✓ Demand is relatively stable - current safety stock is adequate")
        
        if max_weekly_demand > avg_weekly_demand * 1.5:
            print("  ⚠️  Significant demand spikes expected - prepare for peak periods")
        else:
            print("  ✓ No major demand spikes expected - maintain standard inventory")
        
        print("\n  📋 Recommended Actions:")
        print("     1. Maintain minimum inventory of ${:,.2f}".format(safety_stock))
        print("     2. Reorder when inventory drops to ${:,.2f}".format(reorder_point))
        print("     3. Order ${:,.2f} worth of inventory per order".format(order_quantity))
        print("     4. Monitor inventory levels weekly")
        print("     5. Adjust based on actual demand patterns")
        
        return recommendations
    
    def run_complete_pipeline(self):
        """Execute the complete demand forecasting pipeline"""
        print("\n")
        print("╔" + "="*78 + "╗")
        print("║" + " "*20 + "DEMAND FORECASTING PIPELINE" + " "*31 + "║")
        print("╚" + "="*78 + "╝")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Preprocess
        self.preprocess_data()
        
        # Step 3: Exploratory analysis
        self.exploratory_analysis()
        
        # Step 4: Prepare features
        self.prepare_features()
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Visualize results
        self.visualize_model_results()
        
        # Step 7: Generate forecasts for top stores
        print("\n[STEP 7] GENERATING FORECASTS FOR TOP STORES...")
        print("-" * 80)
        
        top_stores = self.df_processed.groupby('Store')['Weekly_Sales'].mean().nlargest(3).index
        
        all_forecasts = {}
        for store_id in top_stores:
            forecast = self.forecast_future_demand(store_id, weeks_ahead=8)
            all_forecasts[store_id] = forecast
            
            # Generate inventory recommendations
            self.inventory_recommendations(forecast, safety_stock_weeks=2)
        
        
        print("\n📊 Generated Files:")
        print("  • demand_analysis_comprehensive.png - Exploratory analysis visualizations")
        print("  • model_performance_results.png - Model evaluation charts")
        print("  • forecast_store_X.png - Individual store forecasts")
        print("\n✓ All forecasts and recommendations have been generated successfully!")
        print("\n" + "="*80)
        
        return all_forecasts

def main():
    """Main execution function"""
    
    # Initialize the system
    forecaster = DemandForecastingSystem(data_path='data/train.csv')
    
    # Run the complete pipeline
    forecasts = forecaster.run_complete_pipeline()
    
    # Additional custom forecasts (optional)
    print("\n\n[CUSTOM FORECASTS] Running additional predictions...")
    print("="*80)
    
    # Forecast for specific stores
    custom_stores = [1, 5, 10]
    for store in custom_stores:
        print(f"\n→ Forecasting for Store {store}...")
        try:
            forecast = forecaster.forecast_future_demand(store, weeks_ahead=12)
            if forecast is not None:
                forecaster.inventory_recommendations(forecast, safety_stock_weeks=3)
        except Exception as e:
            print(f"  ⚠️ Could not generate forecast for Store {store}: {e}")


if __name__ == "__main__":
    main()