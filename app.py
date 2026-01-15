import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Custom CSS
def show_floating_banner(message, color="#28a745"):
    """
    Displays a rounded floating banner at the top that disappears after 1.5 seconds.
    """
    st.markdown(f"""
        <style>
        .floating-banner {{
            position: fixed;
            top: 20px;
            left: 50%;
            /* Centering applied here for the static state */
            transform: translateX(-50%); 
            
            width: auto;
            min-width: 300px;
            max-width: 80%;
            background-color: {color};
            color: white;
            text-align: center;
            padding: 10px 25px;
            z-index: 999999;
            font-weight: bold;
            border-radius: 30px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
            font-family: 'Source Sans Pro', sans-serif;
            animation: fadeOut 1.5s forwards;
            pointer-events: none;
        }}

        @keyframes fadeOut {{
            /* We must include BOTH X and Y translations in every step */
            0% {{ 
                opacity: 1; 
                transform: translate(-50%, 0); 
            }}
            70% {{ 
                opacity: 1; 
                transform: translate(-50%, 0); 
            }}
            100% {{ 
                opacity: 0; 
                transform: translate(-50%, -20px); 
            }}
        }}
        </style>
        <div class="floating-banner">
            {message}
        </div>
    """, unsafe_allow_html=True)

if st.session_state.get('show_train_success_banner', False):
    show_floating_banner("All models trained successfully!", color="#28a745")
    st.session_state.show_train_success_banner = False

# Page configuration
st.set_page_config(page_title="Smoker Status Prediction", layout="wide")

# Title
st.title("Smoker Status Prediction - ML Models")
st.markdown("---")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'confusion_matrices' not in st.session_state:
    st.session_state.confusion_matrices = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    if y_pred_proba is not None:
        metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['AUC Score'] = None
    
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['MCC Score'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


def preprocess_data(X, scaler=None, fit=False):
    X_processed = X.copy()
    
    
    scaling_features = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'systolic', 'relaxation', 'hemoglobin']
    
    # During Inference, I use Fit as false. so It uses the pickled scaler.
    if fit:
        scaler = StandardScaler()
        X_processed[scaling_features] = scaler.fit_transform(X_processed[scaling_features])
    else:
        X_processed[scaling_features] = scaler.transform(X_processed[scaling_features])
    
    
    log_features = ['AST', 'ALT', 'Gtp', 'fasting blood sugar', 'triglyceride']
    for feature in log_features:
        X_processed[feature] = np.log1p(X_processed[feature])
    
    # Map Hearing columns: 1 -> 0, 2 -> 1
    X_processed['hearing(left)'] = X_processed['hearing(left)'].map({1: 0, 2: 1})
    X_processed['hearing(right)'] = X_processed['hearing(right)'].map({1: 0, 2: 1})
    
    return X_processed, scaler

# ==================== SECTION 1: DATASET SELECTION ====================
st.header("Section 1: Dataset Selection")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.trained = False
            st.session_state.metrics_data = None
            st.session_state.confusion_matrices = None
            st.session_state.last_uploaded_file = uploaded_file.name
        
        df = pd.read_csv(uploaded_file)

        if not st.session_state.trained:
            show_floating_banner(f"Loaded: {uploaded_file.name} | Shape: {df.shape}")
        
        with st.expander("Preview Dataset"):
            st.dataframe(df.head(10))

with col2:
    st.write("")
    st.write("")
    train_button = st.button("Train Models", type="primary", use_container_width=True)

st.markdown("---")

# ==================== TRAINING LOGIC ====================
if train_button and uploaded_file is not None:
    with st.spinner("Training models... Please wait..."):
        try:
            X = df.drop('smoking', axis=1)
            y = df['smoking']

            scaler = joblib.load('models/scaler.pkl')
            
            
            X_processed, _ = preprocess_data(X, scaler, fit=False)

            models_info = [
                ("Logistic Regression", "models/logistic_regression_model.pkl"),
                ("Decision Tree", "models/decision_tree_model.pkl"),
                ("K-Nearest Neighbors", "models/knn_model.pkl"),
                ("Gaussian Naive Bayes", "models/gaussian_nb_model.pkl"),
                ("Random Forest", "models/random_forest_model.pkl"),
                ("XGBoost", "models/xgboost_model.pkl")
            ]
            
            # Store metrics and confusion matrices
            all_metrics = {}
            all_cm = {}
            
            # Evaluate each model
            for model_name, model_path in models_info:
                model = joblib.load(model_path)
                y_pred = model.predict(X_processed)
                y_pred_proba = model.predict_proba(X_processed)[:, 1]
                
                # Calculate metrics
                metrics = calculate_metrics(y, y_pred, y_pred_proba)
                all_metrics[model_name] = metrics
                
                # Calculate confusion matrix
                cm = confusion_matrix(y, y_pred)
                all_cm[model_name] = cm
            
            # Store in session state
            st.session_state.trained = True
            st.session_state.metrics_data = all_metrics
            st.session_state.confusion_matrices = all_cm
            st.session_state.show_train_success_banner = True 
            uploaded_file = None
            st.rerun()
            
            show_floating_banner("All models trained successfully!", color="#ff3333")
            st.rerun()
            
        except Exception as e:
            show_floating_banner(f"Error: {str(e)}")

# ==================== SECTION 2: MODEL PERFORMANCE ====================
st.header("Section 2: Model Performance")

if not st.session_state.trained:
    st.info("No data found. Please upload a dataset and click 'Train Models' to view results.")
else:
    # Model selection dropdown
    model_names = list(st.session_state.metrics_data.keys())
    selected_model = st.selectbox(
        "Select a Model to View Performance:",
        model_names,
        index=0
    )
    
    st.markdown("---")
    
    # Get data for selected model
    metrics = st.session_state.metrics_data[selected_model]
    cm = st.session_state.confusion_matrices[selected_model]
    
    
    st.subheader(f"{selected_model} - Performance Metrics")
    
    
    col1, col2 = st.columns([1, 1])
    
    # Left column: Metrics
    with col1:
        st.markdown("### Evaluation Metrics")
        
        
        st.markdown("""
        <style>
        .dataframe {
            font-size: 16px !important;
        }
        .dataframe td {
            font-size: 16px !important;
            padding: 12px !important;
        }
        .dataframe th {
            font-size: 17px !important;
            padding: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Score': [f"{v:.4f}" if v is not None else "N/A" for v in metrics.values()]
        })
        
        
        st.table(metrics_df)
        

        st.markdown("### Key Metrics")
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            if metrics['Accuracy'] is not None:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        
        with metric_cols[1]:
            if metrics['F1 Score'] is not None:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        
        with metric_cols[2]:
            if metrics['AUC Score'] is not None:
                st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    
    # Right column: Confusion Matrix
    with col2:
        st.markdown("### Confusion Matrix")
        
        # Create heatmap using plotly
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted: Non-Smoker', 'Predicted: Smoker'],
            y=['Actual: Non-Smoker', 'Actual: Smoker'],
            colorscale='Blues',
            showscale=True,
            annotation_text=cm
        )
        
        fig.update_layout(
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="Actual Label"),
            height=400,
            font=dict(size=12)
        )
        
        
        for annotation in fig.layout.annotations:
            annotation.font.size = 16
            annotation.font.color = "white" if int(annotation.text) > cm.max() / 2 else "black"
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown("#### Confusion Matrix Breakdown:")
        breakdown_cols = st.columns(2)
        
        with breakdown_cols[0]:
            st.metric("True Positives (TP)", tp)
            st.metric("False Positives (FP)", fp)
        
        with breakdown_cols[1]:
            st.metric("True Negatives (TN)", tn)
            st.metric("False Negatives (FN)", fn)

st.markdown("---")
st.markdown("**Created by Mohamed Niyaz M** | Machine Learning Assignment")
