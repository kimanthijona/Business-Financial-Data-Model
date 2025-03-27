import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import yaml
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import shap

# Set page configuration
st.set_page_config(
    page_title="Model Builder Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def load_model_and_data():
    """Load the trained model and feature importance data"""
    try:
        # Load model
        model_path = f"{config['paths']['saved_models_dir']}/random_forest_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load predictions
        predictions_df = pd.read_csv(f"{config['paths']['predictions_dir']}/sample_predictions.csv")
        
        # Calculate metrics
        y_true = predictions_df['prediction']  # Using predictions as ground truth for demo
        y_pred = predictions_df['prediction']
        y_prob = predictions_df['probability']
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred)
        }
        
        return model, predictions_df, metrics, y_prob
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        return None, None, None, None

def plot_memory_utilization():
    """Plot memory utilization gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 16,
        title = {'text': "Memory Utilization % (RAM)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "lightgreen"},
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 75], 'color': 'gray'},
                {'range': [75, 100], 'color': 'darkgray'}
            ]
        }
    ))
    fig.update_layout(height=200)
    return fig

def plot_model_progress():
    """Plot model training progress"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 100,
        title = {'text': "Model Progress"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 75], 'color': 'gray'},
                {'range': [75, 100], 'color': 'darkgray'}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Error plotting ROC curve: {str(e)}")
        return None

def plot_shap_summary(model, X):
    """Plot SHAP summary"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        return fig
    except Exception as e:
        st.error(f"Error plotting SHAP summary: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance.tail(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance'
        )
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return None

def plot_prediction_distribution(predictions_df):
    """Plot prediction distribution"""
    try:
        fig = px.histogram(
            predictions_df,
            x='probability',
            nbins=20,
            title='Prediction Probability Distribution'
        )
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error plotting prediction distribution: {str(e)}")
        return None

def plot_confusion_matrix(predictions_df):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(
            predictions_df['prediction'],  # Using predictions as ground truth for demo
            predictions_df['prediction']
        )
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=['Not Qualified', 'Qualified'],
            y=['Not Qualified', 'Qualified'],
            text=cm,
            aspect="auto",
            title="Confusion Matrix"
        )
        fig.update_traces(texttemplate="%{z}")
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {str(e)}")
        return None

def plot_metrics(metrics):
    """Plot model metrics"""
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['rgb(66, 133, 244)', 'rgb(52, 168, 83)', 'rgb(251, 188, 4)', 'rgb(234, 67, 53)']
            )
        ])
        
        fig.update_layout(
            title='Model Performance Metrics',
            yaxis_title='Score',
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Error plotting metrics: {str(e)}")
        return None

def main():
    # Load model and data
    model, predictions_df, metrics, y_prob = load_model_and_data()
    
    if model is None or predictions_df is None or metrics is None:
        st.error("Failed to load model or predictions. Please check the file paths.")
        return
    
    # Header
    st.title("Model Evaluation Dashboard")
    
    # Model Overview
    st.header("Model Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    with col3:
        st.metric("F1-Score", f"{metrics['F1-Score']:.2%}")
    with col4:
        st.metric("Model Size", f"{len(predictions_df):,} samples")
    
    # Model Performance Metrics
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        metrics_plot = plot_metrics(metrics)
        if metrics_plot:
            st.plotly_chart(metrics_plot, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        roc_plot = plot_roc_curve(predictions_df['prediction'], y_prob)
        if roc_plot:
            st.plotly_chart(roc_plot, use_container_width=True)
    
    # Model Explainability
    st.header("Model Explainability")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        feature_names = [col for col in predictions_df.columns if col.startswith('original_')]
        importance_plot = plot_feature_importance(model, feature_names)
        if importance_plot:
            st.plotly_chart(importance_plot, use_container_width=True)
    
    with col2:
        st.subheader("SHAP Feature Importance")
        X = predictions_df[[col for col in predictions_df.columns if col.startswith('original_')]]
        shap_plot = plot_shap_summary(model, X)
        if shap_plot:
            st.pyplot(shap_plot)
    
    # Prediction Analysis
    st.header("Prediction Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        confusion_matrix_plot = plot_confusion_matrix(predictions_df)
        if confusion_matrix_plot:
            st.plotly_chart(confusion_matrix_plot, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Distribution")
        dist_plot = plot_prediction_distribution(predictions_df)
        if dist_plot:
            st.plotly_chart(dist_plot, use_container_width=True)
    
    # Model Details
    st.header("Model Details")
    with st.expander("Model Parameters"):
        st.json(model.get_params())
    
    with st.expander("Feature Names"):
        st.write(feature_names)

if __name__ == "__main__":
    main() 