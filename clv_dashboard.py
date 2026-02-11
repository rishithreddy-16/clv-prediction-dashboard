"""
CUSTOMER LIFETIME VALUE PREDICTION DASHBOARD
=============================================
Interactive Streamlit app for CLV prediction and customer segmentation

Run with: streamlit run clv_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .st-emotion-cache-16idsys p {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# DATA LOADING & MODEL TRAINING FUNCTIONS
# ============================================

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Online Retail dataset"""
    
    try:
        # Try to load from UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        df = pd.read_excel(url)
    except:
        st.error("⚠️ Could not download dataset. Please download 'Online Retail.xlsx' manually and place it in the same folder.")
        st.stop()
    
    # Clean data
    df = df[df['CustomerID'].notna()]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Remove outliers
    revenue_99th = df['Revenue'].quantile(0.99)
    df = df[df['Revenue'] <= revenue_99th]
    
    return df

@st.cache_data
def create_features_and_target(_df):
    """Create features and target with time-based split"""
    
    # Time split
    min_date = _df['InvoiceDate'].min()
    max_date = _df['InvoiceDate'].max()
    date_range = (max_date - min_date).days
    split_date = min_date + pd.Timedelta(days=int(date_range * 0.75))
    
    df_train = _df[_df['InvoiceDate'] < split_date]
    df_test = _df[_df['InvoiceDate'] >= split_date]
    
    # Feature engineering
    reference_date = split_date
    
    customer_features = df_train.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': ['sum', 'mean', 'std', 'max', 'min']
    }).reset_index()
    
    customer_features.columns = [
        'CustomerID', 'Recency', 'Frequency', 
        'TotalRevenue', 'AvgRevenue', 'StdRevenue', 'MaxRevenue', 'MinRevenue'
    ]
    
    customer_features['StdRevenue'] = customer_features['StdRevenue'].fillna(0)
    customer_features['AvgDaysBetweenPurchases'] = customer_features['Recency'] / customer_features['Frequency']
    customer_features['RevenuePerOrder'] = customer_features['TotalRevenue'] / customer_features['Frequency']
    
    # Basket size
    basket_size = df_train.groupby(['CustomerID', 'InvoiceNo'])['Quantity'].sum().groupby('CustomerID').mean()
    customer_features = customer_features.merge(
        basket_size.rename('AvgBasketSize').reset_index(),
        on='CustomerID',
        how='left'
    )
    
    # Unique products
    unique_products = df_train.groupby('CustomerID')['StockCode'].nunique().rename('UniqueProducts')
    customer_features = customer_features.merge(
        unique_products.reset_index(),
        on='CustomerID',
        how='left'
    )
    
    # Target: Future CLV
    future_clv = df_test.groupby('CustomerID')['Revenue'].sum().rename('FutureCLV')
    
    # Merge
    final_df = customer_features.merge(future_clv, on='CustomerID', how='inner')
    
    feature_cols = [col for col in final_df.columns if col not in ['CustomerID', 'FutureCLV']]
    
    return final_df, feature_cols, split_date

@st.cache_resource
def train_model(_X, _y):
    """Train Gradient Boosting model"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# ============================================
# LOAD DATA
# ============================================

with st.spinner('🔄 Loading data and training model...'):
    df_raw = load_and_prepare_data()
    final_df, feature_cols, split_date = create_features_and_target(df_raw)
    
    X = final_df[feature_cols]
    y = final_df['FutureCLV']
    
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)
    
    # Metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# ============================================
# SIDEBAR
# ============================================

st.sidebar.image("https://img.icons8.com/color/96/000000/money-bag.png", width=100)
st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["📊 Overview", "🔮 Predictions", "👥 Customer Segments", "📈 Model Performance", "🎮 Try It Yourself"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    """
    **CLV Prediction System**
    
    Predicts future customer lifetime value using machine learning.
    
    - 📦 Dataset: Online Retail
    - 🤖 Model: Gradient Boosting
    - 🎯 Accuracy: R² = {:.2f}
    - 📊 Customers: {:,}
    """.format(r2_test, len(final_df))
)

# ============================================
# PAGE 1: OVERVIEW
# ============================================

if page == "📊 Overview":
    st.title("💰 Customer Lifetime Value Prediction Dashboard")
    st.markdown("### Predict future customer value to optimize marketing ROI")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(final_df):,}",
            delta=f"{len(final_df[final_df['FutureCLV'] > 1000]):,} VIP"
        )
    
    with col2:
        st.metric(
            label="Avg Future CLV",
            value=f"${y.mean():.2f}",
            delta=f"±${mae_test:.0f} error"
        )
    
    with col3:
        st.metric(
            label="Model R² Score",
            value=f"{r2_test:.2%}",
            delta="Good" if r2_test > 0.4 else "Needs work"
        )
    
    with col4:
        st.metric(
            label="Total Revenue at Stake",
            value=f"${y.sum()/1000:.0f}K",
            delta="Next 3 months"
        )
    
    st.markdown("---")
    
    # CLV Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 CLV Distribution")
        fig = px.histogram(
            final_df, 
            x='FutureCLV',
            nbins=50,
            title="Distribution of Future Customer Lifetime Value",
            labels={'FutureCLV': 'Future CLV ($)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Top 10 Highest Value Customers")
        top_customers = final_df.nlargest(10, 'FutureCLV')[['CustomerID', 'FutureCLV']].reset_index(drop=True)
        top_customers.index = top_customers.index + 1
        top_customers['FutureCLV'] = top_customers['FutureCLV'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_customers, use_container_width=True, height=400)
    
    # Feature importance
    st.markdown("---")
    st.markdown("#### 🔍 What Drives Customer Value?")
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(8)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 8 Most Important Features",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: PREDICTIONS
# ============================================

elif page == "🔮 Predictions":
    st.title("🔮 Model Predictions")
    st.markdown("### See how the model performs on test customers")
    
    # Prediction accuracy
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2_test:.3f}", "Test Set")
    with col2:
        st.metric("Mean Absolute Error", f"${mae_test:.2f}")
    with col3:
        st.metric("RMSE", f"${rmse_test:.2f}")
    
    st.markdown("---")
    
    # Actual vs Predicted scatter
    st.markdown("#### 📈 Actual vs Predicted CLV")
    
    prediction_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_test,
        'Error': y_pred_test - y_test,
        'Error_Pct': ((y_pred_test - y_test) / y_test * 100)
    })
    
    fig = px.scatter(
        prediction_df,
        x='Actual',
        y='Predicted',
        title="Model Predictions vs Actual Values",
        labels={'Actual': 'Actual CLV ($)', 'Predicted': 'Predicted CLV ($)'},
        opacity=0.6,
        color='Error_Pct',
        color_continuous_scale='RdYlGn_r',
        hover_data=['Error']
    )
    
    # Perfect prediction line
    max_val = max(prediction_df['Actual'].max(), prediction_df['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error distribution
    st.markdown("---")
    st.markdown("#### 📊 Prediction Error Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            prediction_df,
            x='Error',
            nbins=50,
            title="Distribution of Prediction Errors",
            labels={'Error': 'Prediction Error ($)'},
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            prediction_df,
            y='Error_Pct',
            title="Error Percentage Box Plot",
            labels={'Error_Pct': 'Error %'},
            color_discrete_sequence=['#2ca02c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample predictions table
    st.markdown("---")
    st.markdown("#### 🔢 Sample Predictions")
    
    sample_df = prediction_df.sample(20).copy()
    sample_df['Actual'] = sample_df['Actual'].apply(lambda x: f"${x:,.2f}")
    sample_df['Predicted'] = sample_df['Predicted'].apply(lambda x: f"${x:,.2f}")
    sample_df['Error'] = sample_df['Error'].apply(lambda x: f"${x:,.2f}")
    sample_df['Error_Pct'] = sample_df['Error_Pct'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(sample_df, use_container_width=True)

# ============================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.markdown("### Identify high-value customers for targeted marketing")
    
    # Add predicted CLV to dataframe
    final_df['PredictedCLV'] = y_pred_all
    
    # Segment customers
    final_df['Segment'] = pd.cut(
        final_df['PredictedCLV'],
        bins=[0, 100, 500, 1000, float('inf')],
        labels=['Low (<$100)', 'Medium ($100-500)', 'High ($500-1K)', 'VIP (>$1K)']
    )
    
    # Segment stats
    segment_stats = final_df.groupby('Segment').agg({
        'CustomerID': 'count',
        'PredictedCLV': 'mean',
        'FutureCLV': 'sum'
    }).reset_index()
    
    segment_stats.columns = ['Segment', 'Count', 'Avg Predicted CLV', 'Total Actual Revenue']
    segment_stats['Percentage'] = (segment_stats['Count'] / segment_stats['Count'].sum() * 100).round(1)
    
    # Segment overview
    st.markdown("#### 📊 Segment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            segment_stats,
            values='Count',
            names='Segment',
            title="Customer Distribution by Segment",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_stats,
            x='Segment',
            y='Total Actual Revenue',
            title="Revenue by Segment",
            color='Segment',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment table
    st.markdown("---")
    st.markdown("#### 📋 Segment Statistics")
    
    display_stats = segment_stats.copy()
    display_stats['Avg Predicted CLV'] = display_stats['Avg Predicted CLV'].apply(lambda x: f"${x:,.2f}")
    display_stats['Total Actual Revenue'] = display_stats['Total Actual Revenue'].apply(lambda x: f"${x:,.0f}")
    display_stats['Percentage'] = display_stats['Percentage'].apply(lambda x: f"{x}%")
    
    st.dataframe(display_stats, use_container_width=True, hide_index=True)
    
    # Marketing recommendations
    st.markdown("---")
    st.markdown("#### 💡 Marketing Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **VIP Segment (>$1K)**
        - {} customers ({:.1f}%)
        - Avg value: ${:.0f}
        
        **Actions:**
        - Dedicated account manager
        - Exclusive early access
        - Premium customer service
        - Personalized offers
        """.format(
            segment_stats[segment_stats['Segment'] == 'VIP (>$1K)']['Count'].values[0],
            segment_stats[segment_stats['Segment'] == 'VIP (>$1K)']['Percentage'].values[0],
            segment_stats[segment_stats['Segment'] == 'VIP (>$1K)']['Avg Predicted CLV'].values[0]
        ))
    
    with col2:
        st.info("""
        **Budget Allocation**
        
        Recommended spend:
        - 60% on VIP tier
        - 30% on High tier
        - 10% on Medium/Low tier
        
        **Expected ROI:**
        - VIP: 5-10x
        - High: 3-5x
        - Medium: 1.5-3x
        - Low: 0.5-1.5x
        """)

# ============================================
# PAGE 4: MODEL PERFORMANCE
# ============================================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Analysis")
    st.markdown("### Deep dive into model metrics and diagnostics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2_test:.4f}")
    with col2:
        st.metric("MAE", f"${mae_test:.2f}")
    with col3:
        st.metric("RMSE", f"${rmse_test:.2f}")
    with col4:
        mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        st.metric("MAPE", f"{mape:.1f}%")
    
    st.markdown("---")
    
    # Feature importance detailed
    st.markdown("#### 🔍 Feature Importance Analysis")
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title="All Features Ranked by Importance",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals analysis
    st.markdown("---")
    st.markdown("#### 📉 Residual Analysis")
    
    residuals = y_test - y_pred_test
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            x=y_pred_test,
            y=residuals,
            title="Residuals vs Predicted Values",
            labels={'x': 'Predicted CLV ($)', 'y': 'Residuals ($)'},
            opacity=0.5
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            residuals,
            nbins=50,
            title="Distribution of Residuals",
            labels={'value': 'Residuals ($)'},
            color_discrete_sequence=['#e377c2']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model details
    st.markdown("---")
    st.markdown("#### ⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Type:** Gradient Boosting Regressor
        
        **Hyperparameters:**
        - n_estimators: 100
        - max_depth: 5
        - learning_rate: 0.1 (default)
        - random_state: 42
        """)
    
    with col2:
        st.success("""
        **Dataset Split:**
        - Training: {} customers
        - Testing: {} customers
        - Test size: 20%
        
        **Features:** {} behavioral metrics
        """.format(len(X_train), len(X_test), len(feature_cols)))

# ============================================
# PAGE 5: TRY IT YOURSELF
# ============================================

elif page == "🎮 Try It Yourself":
    st.title("🎮 Predict Customer Lifetime Value")
    st.markdown("### Enter customer characteristics to get a CLV prediction")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Purchase Behavior")
        
        recency = st.slider(
            "Days Since Last Purchase",
            min_value=0,
            max_value=365,
            value=30,
            help="How many days ago did they last buy?"
        )
        
        frequency = st.slider(
            "Number of Orders",
            min_value=1,
            max_value=50,
            value=10,
            help="Total number of orders placed"
        )
        
        total_revenue = st.slider(
            "Total Past Revenue ($)",
            min_value=0,
            max_value=10000,
            value=1000,
            step=100,
            help="Total amount spent historically"
        )
    
    with col2:
        st.markdown("#### 💰 Spending Patterns")
        
        avg_revenue = st.slider(
            "Average Order Value ($)",
            min_value=0,
            max_value=1000,
            value=100,
            step=10
        )
        
        unique_products = st.slider(
            "Unique Products Purchased",
            min_value=1,
            max_value=100,
            value=15
        )
        
        avg_basket = st.slider(
            "Average Items Per Order",
            min_value=1,
            max_value=50,
            value=10
        )
    
    # Calculate derived features
    max_revenue = avg_revenue * 2  # Simple estimation
    min_revenue = avg_revenue * 0.5
    std_revenue = avg_revenue * 0.3
    avg_days_between = recency / frequency if frequency > 0 else recency
    revenue_per_order = total_revenue / frequency if frequency > 0 else total_revenue
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'TotalRevenue': [total_revenue],
        'AvgRevenue': [avg_revenue],
        'StdRevenue': [std_revenue],
        'MaxRevenue': [max_revenue],
        'MinRevenue': [min_revenue],
        'AvgDaysBetweenPurchases': [avg_days_between],
        'RevenuePerOrder': [revenue_per_order],
        'AvgBasketSize': [avg_basket],
        'UniqueProducts': [unique_products]
    })
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Predict CLV", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("### 🎯 Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Future CLV",
                f"${prediction:.2f}",
                delta="Next 3 months"
            )
        
        with col2:
            if prediction < 100:
                segment = "Low Value"
                color = "🔵"
            elif prediction < 500:
                segment = "Medium Value"
                color = "🟢"
            elif prediction < 1000:
                segment = "High Value"
                color = "🟡"
            else:
                segment = "VIP"
                color = "🔴"
            
            st.metric("Customer Segment", f"{color} {segment}")
        
        with col3:
            confidence_interval = mae_test
            st.metric(
                "Confidence Range",
                f"±${confidence_interval:.2f}",
                delta="Likely range"
            )
        
        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 Marketing Recommendation")
        
        if prediction >= 1000:
            st.success("""
            **🔴 VIP CUSTOMER - High Priority**
            
            This customer is predicted to be highly valuable!
            
            **Recommended Actions:**
            - Assign dedicated account manager
            - Offer exclusive VIP perks and early access
            - Provide premium customer support
            - Send personalized product recommendations
            - Budget: Allocate $50-100 for acquisition/retention
            """)
        elif prediction >= 500:
            st.info("""
            **🟡 HIGH VALUE CUSTOMER - Worth Investment**
            
            This customer shows strong potential.
            
            **Recommended Actions:**
            - Enroll in loyalty program
            - Send targeted email campaigns
            - Offer bundle deals and cross-sells
            - Budget: Allocate $20-50 for marketing
            """)
        elif prediction >= 100:
            st.warning("""
            **🟢 MEDIUM VALUE CUSTOMER - Standard Treatment**
            
            This customer has moderate potential.
            
            **Recommended Actions:**
            - Include in general email campaigns
            - Offer standard promotions
            - Monitor for upsell opportunities
            - Budget: Allocate $5-20 for marketing
            """)
        else:
            st.error("""
            **🔵 LOW VALUE CUSTOMER - Minimal Investment**
            
            This customer shows limited potential.
            
            **Recommended Actions:**
            - Automated marketing only
            - Focus on conversion to medium tier
            - Consider churn risk
            - Budget: Allocate <$5 for marketing
            """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: #666;'>
        Built with ❤️ using Streamlit | Data Science Portfolio Project<br>
        Model: Gradient Boosting | Dataset: UCI Online Retail | Accuracy: R² = {:.2%}
    </p>
</div>
""".format(r2_test), unsafe_allow_html=True)