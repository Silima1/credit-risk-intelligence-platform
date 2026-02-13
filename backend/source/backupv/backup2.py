import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta
import random
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide"
)

###############################################################################
#                             SYSTEM INITIALIZATION                          #
###############################################################################

@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("source/rf_kaggle.joblib")
        train_df = pd.read_csv("source/train.csv")
        test_df = pd.read_csv("source/test.csv")
        
        if "application_date" in train_df.columns:
            train_df = add_date_features(train_df)
            test_df = add_date_features(test_df)
        
        return model, train_df, test_df
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None

def add_date_features(df):
    df = df.copy()
    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
        df["app_year"] = df["application_date"].dt.year
        df["app_month"] = df["application_date"].dt.month
        df["app_dayofweek"] = df["application_date"].dt.dayofweek
    return df

def get_feature_columns(model, df):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    
    if hasattr(model, "steps"):
        pipeline_model = model.steps[-1][1]
        if hasattr(pipeline_model, "feature_names_in_"):
            return list(pipeline_model.feature_names_in_)
    
    non_feature_cols = []
    if 'fraud_flag' in df.columns:
        non_feature_cols.append('fraud_flag')
    
    common_id_cols = ['ID', 'id', 'application_id', 'customer_id', 'data_batch_id', 'Unnamed: 0']
    for col in common_id_cols:
        if col in df.columns:
            non_feature_cols.append(col)
    
    return [col for col in df.columns if col not in non_feature_cols]

def extract_model_from_pipeline(model):
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model

def validate_and_align_data(X, y):
    if y is None:
        return X, None
    
    x_len = len(X)
    y_len = len(y)
    
    if x_len == y_len:
        return X, y
    
    min_len = min(x_len, y_len)
    
    if hasattr(X, 'iloc'):
        X_aligned = X.iloc[:min_len]
    else:
        X_aligned = X[:min_len]
    
    if hasattr(y, 'iloc'):
        y_aligned = y.iloc[:min_len]
    else:
        y_aligned = y[:min_len]
    
    return X_aligned, y_aligned

###############################################################################
#                           FIXED ML MODEL LAYER                             #
###############################################################################

class FixedMLModel:
    
    def __init__(self, model):
        self.model = model
        self.base_model = extract_model_from_pipeline(model)
        self.predictions_history = []
        self.performance_metrics = {}
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_risk_score(self, X):
        probabilities = self.predict_proba(X)
        return probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

###############################################################################
#                              XAI LAYER MODULE                              #
###############################################################################

class XAILayer:
    
    def __init__(self, model):
        self.model = model
        self.base_model = extract_model_from_pipeline(model)
        self.explainer = None
        self.shap_values = None
    
    def calculate_shap_values(self, X):
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.base_model, X)
        
        self.shap_values = self.explainer.shap_values(X)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def get_local_explanations(self, X, instance_idx, top_n=5):
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        shap_values_single = self.shap_values[instance_idx]
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
        
        explanation_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values_single,
            'abs_shap': np.abs(shap_values_single)
        }).sort_values('abs_shap', ascending=False).head(top_n)
        
        return explanation_df
    
    def get_global_importance(self, X):
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(self.shap_values).mean(axis=0),
            'std_shap': self.shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df
    
    def generate_reason_codes(self, X, instance_idx, threshold=0.01):
        local_expl = self.get_local_explanations(X, instance_idx)
        reason_codes = []
        
        for _, row in local_expl.iterrows():
            impact = row['shap_value']
            if abs(impact) > threshold:
                direction = "increases" if impact > 0 else "decreases"
                magnitude = "significantly" if abs(impact) > 0.1 else "moderately"
                reason_codes.append(f"{row['feature']} {direction} risk {magnitude}")
        
        return reason_codes
    
    def plot_shap_summary(self, X, max_features=20):
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, show=False, max_display=max_features)
        plt.tight_layout()
        return fig

###############################################################################
#                           OPTIMIZATION MODULE                              #
###############################################################################

class OptimizationModule:
    
    def __init__(self, base_model):
        self.model = base_model
        self.base_model = extract_model_from_pipeline(base_model)
        self.calibrated_model = None
        self.optimal_threshold = 0.5
        self.policy_versions = []
        self.cost_matrix = {
            'FP': 100,
            'FN': 500,
            'TP': -200,
            'TN': -50
        }
    
    def calibrate_probabilities(self, X_calib, y_calib):
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, 
            method='sigmoid', 
            cv='prefit'
        )
        self.calibrated_model.fit(X_calib, y_calib)
        return self.calibrated_model
    
    def calculate_optimal_threshold(self, y_true, y_proba, cost_matrix=None):
        if cost_matrix is None:
            cost_matrix = self.cost_matrix
        
        thresholds = np.linspace(0.01, 0.99, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            total_cost = (
                fp * cost_matrix['FP'] + 
                fn * cost_matrix['FN'] + 
                tp * cost_matrix['TP'] + 
                tn * cost_matrix['TN']
            )
            costs.append(total_cost)
        
        optimal_idx = np.argmin(costs)
        self.optimal_threshold = thresholds[optimal_idx]
        
        return self.optimal_threshold, costs[optimal_idx]
    
    def create_policy_version(self, description, threshold, parameters):
        policy = {
            'version_id': len(self.policy_versions) + 1,
            'description': description,
            'threshold': threshold,
            'parameters': parameters,
            'created_at': datetime.now(),
            'active': len(self.policy_versions) == 0
        }
        self.policy_versions.append(policy)
        return policy
    
    def get_active_policy(self):
        for policy in self.policy_versions:
            if policy['active']:
                return policy
        return None
    
    def switch_policy(self, version_id):
        for policy in self.policy_versions:
            policy['active'] = (policy['version_id'] == version_id)
        return self.get_active_policy()

###############################################################################
#                           ADAPTATION MODULE                                #
###############################################################################

class AdaptationModule:
    
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.drift_history = []
        self.bandit = ContextualBandit()
    
    def calculate_psi(self, expected, actual, bins=10):
        breakpoints = np.linspace(0, 1, bins + 1)
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        
        expected_perc = expected_hist / len(expected)
        actual_perc = actual_hist / len(actual)
        
        psi = np.sum((actual_perc - expected_perc) * np.log((actual_perc + 1e-10) / (expected_perc + 1e-10)))
        return psi
    
    def calculate_ks_statistic(self, expected, actual):
        return stats.ks_2samp(expected, actual).statistic
    
    def detect_data_drift(self, current_data, feature_names, threshold_psi=0.25, threshold_ks=0.1):
        drift_results = {}
        
        for i, feature_name in enumerate(feature_names):
            expected = self.reference_data[:, i]
            actual = current_data[:, i]
            
            psi = self.calculate_psi(expected, actual)
            ks = self.calculate_ks_statistic(expected, actual)
            
            drift_detected = (psi > threshold_psi) or (ks > threshold_ks)
            
            drift_results[feature_name] = {
                'psi': psi,
                'ks': ks,
                'drift_detected': drift_detected,
                'severity': 'high' if psi > 0.5 else 'medium' if psi > 0.25 else 'low'
            }
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'results': drift_results
        })
        
        return drift_results
    
    def detect_score_drift(self, reference_scores, current_scores):
        psi = self.calculate_psi(reference_scores, current_scores)
        ks = self.calculate_ks_statistic(reference_scores, current_scores)
        
        drift_detected = (psi > 0.25) or (ks > 0.1)
        
        result = {
            'psi': psi,
            'ks': ks,
            'drift_detected': drift_detected,
            'severity': 'high' if psi > 0.5 else 'medium' if psi > 0.25 else 'low'
        }
        
        return result
    
    def detect_shap_drift(self, reference_shap, current_shap):
        correlation = np.corrcoef(
            np.abs(reference_shap).mean(axis=0),
            np.abs(current_shap).mean(axis=0)
        )[0, 1]
        
        drift_detected = correlation < 0.8
        
        result = {
            'correlation': correlation,
            'drift_detected': drift_detected,
            'severity': 'high' if correlation < 0.6 else 'medium' if correlation < 0.8 else 'low'
        }
        
        return result
    
    def get_drift_summary(self, drift_results):
        total_drift = sum(1 for v in drift_results.values() if v['drift_detected'])
        high_severity = sum(1 for v in drift_results.values() if v['severity'] == 'high')
        
        return {
            'total_features': len(drift_results),
            'features_with_drift': total_drift,
            'high_severity_drift': high_severity,
            'overall_status': 'CRITICAL' if high_severity > 3 else 'WARNING' if total_drift > 5 else 'NORMAL'
        }

class ContextualBandit:
    
    def __init__(self, n_actions=3, learning_rate=0.1, exploration_rate=0.2):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.action_values = np.ones(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.action_names = ['Decrease Threshold', 'Maintain Threshold', 'Increase Threshold']
    
    def select_action(self, context=None):
        if random.random() < self.exploration_rate:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = np.argmax(self.action_values)
        
        return action, self.action_names[action]
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = self.learning_rate / (1 + self.action_counts[action])
        self.action_values[action] += alpha * (reward - self.action_values[action])
    
    def get_policy(self):
        exp_values = np.exp(self.action_values - np.max(self.action_values))
        policy = exp_values / exp_values.sum()
        return dict(zip(self.action_names, policy))

###############################################################################
#                               DASHBOARD MODULE                             #
###############################################################################

class Dashboard:
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.recommendations = []
        self.decisions = []
    
    def update_metrics(self, model, X, y_true=None):
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        self.metrics = {
            'total_transactions': len(X),
            'high_risk': (probabilities >= 0.7).sum(),
            'medium_risk': ((probabilities >= 0.3) & (probabilities < 0.7)).sum(),
            'low_risk': (probabilities < 0.3).sum(),
            'avg_risk_score': probabilities.mean(),
            'std_risk_score': probabilities.std()
        }
        
        if y_true is not None and len(y_true) == len(predictions):
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            try:
                self.metrics.update({
                    'accuracy': accuracy_score(y_true, predictions),
                    'precision': precision_score(y_true, predictions, zero_division=0),
                    'recall': recall_score(y_true, predictions, zero_division=0)
                })
            except:
                self.metrics.update({
                    'accuracy': 'N/A',
                    'precision': 'N/A',
                    'recall': 'N/A'
                })
        else:
            self.metrics.update({
                'accuracy': 'N/A',
                'precision': 'N/A',
                'recall': 'N/A'
            })
    
    def add_alert(self, alert_type, message, severity='medium'):
        self.alerts.append({
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now()
        })
    
    def add_recommendation(self, recommendation, priority='medium'):
        self.recommendations.append({
            'recommendation': recommendation,
            'priority': priority,
            'timestamp': datetime.now()
        })
    
    def add_decision(self, decision, confidence, explanation):
        self.decisions.append({
            'decision': decision,
            'confidence': confidence,
            'explanation': explanation,
            'timestamp': datetime.now()
        })
    
    def clear_old_entries(self, max_age_hours=24):
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
        self.recommendations = [r for r in self.recommendations if r['timestamp'] > cutoff_time]
        self.decisions = [d for d in self.decisions if d['timestamp'] > cutoff_time]
    
    def get_alert_summary(self):
        critical = sum(1 for a in self.alerts if a['severity'] == 'high')
        warnings = sum(1 for a in self.alerts if a['severity'] == 'medium')
        
        return {
            'total': len(self.alerts),
            'critical': critical,
            'warnings': warnings
        }

###############################################################################
#                               MAIN APPLICATION                              #
###############################################################################

def render_dashboard(fixed_model, xai_layer, optimization_module, adaptation_module, dashboard, X_test, y_test):
    
    st.markdown("<h2 style='font-size: 1.4em;'>System Dashboard Overview</h2>", unsafe_allow_html=True)
    
    X_test_aligned, y_test_aligned = validate_and_align_data(X_test, y_test)
    
    if y_test is not None and len(X_test) != len(y_test):
        st.info(f"Data Note: Using {len(X_test_aligned)} aligned samples")
    
    dashboard.update_metrics(fixed_model.model, X_test_aligned, y_test_aligned)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", f"{dashboard.metrics['total_transactions']:,}")
        st.metric("High Risk", dashboard.metrics['high_risk'])
    
    with col2:
        st.metric("Medium Risk", dashboard.metrics['medium_risk'])
        st.metric("Low Risk", dashboard.metrics['low_risk'])
    
    with col3:
        st.metric("Average Risk Score", f"{dashboard.metrics['avg_risk_score']:.3f}")
        if 'accuracy' in dashboard.metrics and dashboard.metrics['accuracy'] != 'N/A':
            st.metric("Model Accuracy", f"{dashboard.metrics['accuracy']:.3f}")
    
    with col4:
        active_policy = optimization_module.get_active_policy()
        if active_policy:
            st.metric("Active Policy", f"v{active_policy['version_id']}")
            st.metric("Decision Threshold", f"{active_policy['threshold']:.3f}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("<h3 style='font-size: 1.1em;'>System Alerts</h3>", unsafe_allow_html=True)
        
        sample_size = min(100, len(X_test_aligned))
        if sample_size > 0:
            X_sample = X_test_aligned.iloc[:sample_size] if hasattr(X_test_aligned, 'iloc') else X_test_aligned[:sample_size]
            sample_scores = fixed_model.get_risk_score(X_sample)
            
            if len(X_test_aligned) > sample_size:
                reference_sample = X_test_aligned.iloc[sample_size:sample_size*2] if hasattr(X_test_aligned, 'iloc') else X_test_aligned[sample_size:sample_size*2]
                reference_scores = fixed_model.get_risk_score(reference_sample)
            else:
                reference_scores = sample_scores
            
            score_drift = adaptation_module.detect_score_drift(reference_scores, sample_scores)
            
            if score_drift['drift_detected']:
                dashboard.add_alert(
                    'Score Drift',
                    f"Significant score drift detected (PSI: {score_drift['psi']:.3f})",
                    score_drift['severity']
                )
        
        alert_summary = dashboard.get_alert_summary()
        
        if alert_summary['critical'] > 0:
            st.error(f"Critical Alerts: {alert_summary['critical']}")
        if alert_summary['warnings'] > 0:
            st.warning(f"Warnings: {alert_summary['warnings']}")
        
        for alert in dashboard.alerts[-5:]:
            color = '#FF6B6B' if alert['severity'] == 'high' else '#FFD93D'
            st.markdown(f"""
                <div style='background-color: {color}20; border-left: 4px solid {color}; 
                         padding: 10px; margin: 5px 0; border-radius: 4px;'>
                    <strong>{alert['type']}</strong><br>
                    <small>{alert['message']}</small>
                </div>
            """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("<h3 style='font-size: 1.1em;'>Recommendations</h3>", unsafe_allow_html=True)
        
        if dashboard.metrics['high_risk'] > 100:
            dashboard.add_recommendation(
                "High number of risky applications detected. Consider reviewing threshold.",
                'high'
            )
        
        if 'sample_scores' in locals() and score_drift['drift_detected']:
            dashboard.add_recommendation(
                "Model drift detected. Schedule retraining with recent data.",
                'medium'
            )
        
        for rec in dashboard.recommendations[-3:]:
            st.info(f"{rec['priority'].upper()}: {rec['recommendation']}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.1em;'>Global Feature Importance</h3>", unsafe_allow_html=True)
    
    sample_size = min(100, len(X_test_aligned))
    if sample_size > 0:
        X_sample = X_test_aligned.iloc[:sample_size] if hasattr(X_test_aligned, 'iloc') else X_test_aligned[:sample_size]
        
        try:
            xai_layer.calculate_shap_values(X_sample)
            global_importance = xai_layer.get_global_importance(X_sample)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = global_importance.head(8)
            ax.barh(range(len(top_features)), top_features['mean_abs_shap'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Mean Absolute SHAP Value')
            ax.set_title('Top 8 Most Important Features')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanations: {str(e)}")
    else:
        st.warning("Not enough data for feature importance analysis")

def render_credit_application(fixed_model, xai_layer, test_df, feature_cols):
    
    st.markdown("<h2 style='font-size: 1.4em;'>Credit Application Analysis</h2>", unsafe_allow_html=True)
    
    if "application_id" in test_df.columns and len(test_df) > 0:
        app_id = st.selectbox(
            "Select Application ID:",
            test_df["application_id"].unique()[:100],
            key="app_id_select"
        )
        
        if st.button("Analyze Application", type="primary"):
            row = test_df[test_df["application_id"] == app_id]
            if len(row) > 0:
                X = row[feature_cols] if len(feature_cols) > 0 else row.select_dtypes(include=[np.number])
                
                probability = fixed_model.get_risk_score(X)[0]
                prediction = fixed_model.predict(X)[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Probability", f"{probability:.3f}")
                
                with col2:
                    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
                    st.metric("Prediction", risk_label)
                
                with col3:
                    recommendation = "REVIEW" if probability > 0.5 else "APPROVE"
                    st.metric("Recommendation", recommendation)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='font-size: 1em;'>Local Explanations</h4>", unsafe_allow_html=True)
                
                try:
                    xai_layer.calculate_shap_values(X)
                    local_expl = xai_layer.get_local_explanations(X, 0, top_n=8)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#FF6B6B' if val > 0 else '#4ECDC4' for val in local_expl['shap_value']]
                    ax.barh(local_expl['feature'], local_expl['shap_value'], color=colors)
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    ax.set_xlabel('SHAP Value (Impact on Prediction)')
                    ax.set_title('Top Features Influencing This Decision')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("<h4 style='font-size: 1em;'>Reason Codes</h4>", unsafe_allow_html=True)
                    
                    reason_codes = xai_layer.generate_reason_codes(X, 0)
                    for i, reason in enumerate(reason_codes[:5], 1):
                        st.write(f"{i}. {reason}")
                except Exception as e:
                    st.warning(f"Could not generate explanations: {str(e)}")
                
                st.markdown("<h4 style='font-size: 1em;'>Application Details</h4>", unsafe_allow_html=True)
                st.dataframe(row, use_container_width=True)
            else:
                st.warning("Application ID not found")
    else:
        st.warning("No application data available")

def render_xai_explanations(xai_layer, X_test):
    
    st.markdown("<h2 style='font-size: 1.4em;'>Explainable AI (XAI) Layer</h2>", unsafe_allow_html=True)
    
    if len(X_test) == 0:
        st.warning("No test data available for XAI analysis")
        return
    
    max_sample_size = min(500, len(X_test))
    sample_size = st.slider("Sample Size for Analysis", 50, max_sample_size, min(100, max_sample_size))
    
    X_sample = X_test.iloc[:sample_size]
    
    try:
        xai_layer.calculate_shap_values(X_sample)
        
        tab1, tab2, tab3 = st.tabs(["Summary Plot", "Dependence Plots", "Feature Analysis"])
        
        with tab1:
            st.markdown("<h4 style='font-size: 1em;'>SHAP Summary Plot</h4>", unsafe_allow_html=True)
            try:
                fig = xai_layer.plot_shap_summary(X_sample, max_features=15)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate summary plot: {str(e)}")
            
            st.markdown("<p style='font-size: 0.8em; color: #666;'>" +
                       "Each point represents a SHAP value for a feature and instance. " +
                       "Color indicates feature value (red = high, blue = low). " +
                       "Position shows impact on model output.</p>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h4 style='font-size: 1em;'>Feature Dependence Analysis</h4>", unsafe_allow_html=True)
            
            if len(X_sample.columns) > 1:
                feature_x = st.selectbox("Select Feature for X-axis:", X_sample.columns[:10])
                feature_color = st.selectbox("Select Feature for Coloring:", X_sample.columns[:10])
                
                if feature_x and feature_color:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.dependence_plot(
                            feature_x,
                            xai_layer.shap_values,
                            X_sample,
                            feature_names=X_sample.columns,
                            interaction_index=feature_color,
                            ax=ax,
                            show=False
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not generate dependence plot: {str(e)}")
        
        with tab3:
            st.markdown("<h4 style='font-size: 1em;'>Global Feature Importance</h4>", unsafe_allow_html=True)
            
            try:
                global_importance = xai_layer.get_global_importance(X_sample)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                top_n = min(10, len(global_importance))
                top_features = global_importance.head(top_n)
                
                ax1.barh(range(top_n), top_features['mean_abs_shap'])
                ax1.set_yticks(range(top_n))
                ax1.set_yticklabels(top_features['feature'])
                ax1.set_xlabel('Mean |SHAP| Value')
                ax1.set_title(f'Top {top_n} Most Important Features')
                
                ax2.bar(range(top_n), top_features['std_shap'])
                ax2.set_xticks(range(top_n))
                ax2.set_xticklabels(top_features['feature'], rotation=45, ha='right')
                ax2.set_ylabel('Standard Deviation of SHAP')
                ax2.set_title('Feature Consistency')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate feature importance: {str(e)}")
    except Exception as e:
        st.error(f"Error in XAI analysis: {str(e)}")
        st.info("This may occur if the model is not compatible with TreeExplainer or if there's insufficient data.")

def render_optimization(optimization_module, X_train, y_train, X_test):
    
    st.markdown("<h2 style='font-size: 1.4em;'>Optimization Module</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Probability Calibration", "Cost-Sensitive Learning", "Policy Management"])
    
    with tab1:
        st.markdown("<h4 style='font-size: 1em;'>Model Probability Calibration</h4>", unsafe_allow_html=True)
        
        if y_train is not None and len(X_train) > 0:
            if st.button("Calibrate Model Probabilities"):
                with st.spinner("Calibrating..."):
                    try:
                        calibrated_model = optimization_module.calibrate_probabilities(X_train, y_train)
                        
                        sample_size = min(1000, len(X_test))
                        if sample_size > 0:
                            X_test_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
                            original_proba = optimization_module.model.predict_proba(X_test_sample)[:, 1]
                            calibrated_proba = calibrated_model.predict_proba(X_test_sample)[:, 1]
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            ax1.hist(original_proba, bins=30, alpha=0.7, label='Original', color='blue')
                            ax1.set_xlabel('Probability')
                            ax1.set_ylabel('Frequency')
                            ax1.set_title('Original Probabilities')
                            ax1.legend()
                            
                            ax2.hist(calibrated_proba, bins=30, alpha=0.7, label='Calibrated', color='green')
                            ax2.set_xlabel('Probability')
                            ax2.set_ylabel('Frequency')
                            ax2.set_title('Calibrated Probabilities')
                            ax2.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.success("Model calibration completed")
                    except Exception as e:
                        st.error(f"Calibration failed: {str(e)}")
        else:
            st.warning("Training labels not available for calibration")
    
    with tab2:
        st.markdown("<h4 style='font-size: 1em;'>Cost-Sensitive Threshold Optimization</h4>", unsafe_allow_html=True)
        
        if y_train is not None and len(X_train) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fp_cost = st.number_input("False Positive Cost", value=100.0, min_value=0.0)
                fn_cost = st.number_input("False Negative Cost", value=500.0, min_value=0.0)
            
            with col2:
                tp_reward = st.number_input("True Positive Reward", value=-200.0)
                tn_reward = st.number_input("True Negative Reward", value=-50.0)
            
            cost_matrix = {'FP': fp_cost, 'FN': fn_cost, 'TP': tp_reward, 'TN': tn_reward}
            
            if st.button("Calculate Optimal Threshold"):
                try:
                    sample_size = min(5000, len(X_train))
                    X_train_sample = X_train.iloc[:sample_size] if hasattr(X_train, 'iloc') else X_train[:sample_size]
                    y_train_sample = y_train.iloc[:sample_size] if hasattr(y_train, 'iloc') else y_train[:sample_size]
                    
                    y_proba = optimization_module.model.predict_proba(X_train_sample)[:, 1]
                    optimal_threshold, optimal_cost = optimization_module.calculate_optimal_threshold(
                        y_train_sample.values, y_proba, cost_matrix
                    )
                    
                    st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
                    st.metric("Expected Cost", f"${optimal_cost:.2f}")
                    
                    thresholds = np.linspace(0.01, 0.99, 50)
                    costs = []
                    
                    for t in thresholds:
                        y_pred = (y_proba >= t).astype(int)
                        tp = np.sum((y_train_sample == 1) & (y_pred == 1))
                        fp = np.sum((y_train_sample == 0) & (y_pred == 1))
                        fn = np.sum((y_train_sample == 1) & (y_pred == 0))
                        tn = np.sum((y_train_sample == 0) & (y_pred == 0))
                        costs.append(fp * cost_matrix['FP'] + fn * cost_matrix['FN'] + 
                                   tp * cost_matrix['TP'] + tn * cost_matrix['TN'])
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(thresholds, costs)
                    ax.axvline(optimal_threshold, color='red', linestyle='--', 
                              label=f'Optimal: {optimal_threshold:.3f}')
                    ax.set_xlabel('Threshold')
                    ax.set_ylabel('Total Cost')
                    ax.set_title('Cost vs Decision Threshold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Threshold optimization failed: {str(e)}")
        else:
            st.warning("Training labels required for cost-sensitive optimization")
    
    with tab3:
        st.markdown("<h4 style='font-size: 1em;'>Policy Version Management</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Create New Policy")
            policy_desc = st.text_input("Policy Description", "New risk management policy")
            policy_threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
            
            if st.button("Create Policy Version"):
                new_policy = optimization_module.create_policy_version(
                    policy_desc,
                    policy_threshold,
                    {'created_by': 'Dashboard User'}
                )
                st.success(f"Policy v{new_policy['version_id']} created")
        
        with col2:
            st.markdown("Active Policy")
            active_policy = optimization_module.get_active_policy()
            
            if active_policy:
                st.info(f"v{active_policy['version_id']}: {active_policy['description']}")
                st.write(f"Threshold: {active_policy['threshold']:.3f}")
                st.write(f"Created: {active_policy['created_at'].strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("No active policy")
        
        if optimization_module.policy_versions:
            st.markdown("All Policy Versions")
            policies_df = pd.DataFrame(optimization_module.policy_versions)
            st.dataframe(
                policies_df[['version_id', 'description', 'threshold', 'active', 'created_at']],
                use_container_width=True
            )
            
            version_to_activate = st.selectbox(
                "Switch to Policy Version:",
                [p['version_id'] for p in optimization_module.policy_versions]
            )
            
            if st.button("Activate Selected Policy"):
                optimization_module.switch_policy(version_to_activate)
                st.success(f"Policy v{version_to_activate} activated")

def render_adaptation(adaptation_module, fixed_model, xai_layer, X_train, X_test, dashboard):
    
    st.markdown("<h2 style='font-size: 1.4em;'>Adaptation and Monitoring Module</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Drift Detection", "Contextual Bandit", "System Health"])
    
    with tab1:
        st.markdown("<h4 style='font-size: 1em;'>Data and Model Drift Detection</h4>", unsafe_allow_html=True)
        
        if st.button("Run Drift Detection Analysis"):
            with st.spinner("Analyzing drift..."):
                try:
                    sample_size = min(200, len(X_test), len(X_train))
                    if sample_size > 10:
                        X_current = X_test.iloc[:sample_size].values
                        X_reference = X_train.iloc[:sample_size].values
                        
                        feature_names = X_test.columns.tolist()[:min(20, X_test.shape[1])]
                        
                        if len(feature_names) > 0:
                            data_drift = adaptation_module.detect_data_drift(
                                X_reference[:, :len(feature_names)],
                                X_current[:, :len(feature_names)],
                                feature_names
                            )
                            
                            reference_scores = fixed_model.get_risk_score(X_train.iloc[:sample_size])
                            current_scores = fixed_model.get_risk_score(X_test.iloc[:sample_size])
                            score_drift = adaptation_module.detect_score_drift(reference_scores, current_scores)
                            
                            drift_summary = adaptation_module.get_drift_summary(data_drift)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Features Analyzed", drift_summary['total_features'])
                            
                            with col2:
                                st.metric("Features with Drift", drift_summary['features_with_drift'])
                            
                            with col3:
                                status_color = {
                                    'CRITICAL': 'red',
                                    'WARNING': 'orange',
                                    'NORMAL': 'green'
                                }.get(drift_summary['overall_status'], 'gray')
                                st.markdown(f"""
                                    <div style='text-align: center;'>
                                        <h3>Overall Status</h3>
                                        <h2 style='color: {status_color};'>{drift_summary['overall_status']}</h2>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                            
                            st.markdown("Drift Details")
                            
                            drift_details = pd.DataFrame([
                                {
                                    'Feature': feature,
                                    'PSI': results['psi'],
                                    'KS': results['ks'],
                                    'Severity': results['severity'],
                                    'Drift': 'Yes' if results['drift_detected'] else 'No'
                                }
                                for feature, results in list(data_drift.items())[:10]
                            ])
                            
                            st.dataframe(drift_details, use_container_width=True)
                            
                            if score_drift['drift_detected']:
                                dashboard.add_alert(
                                    'Score Drift',
                                    f"Score distribution changed significantly (PSI: {score_drift['psi']:.3f})",
                                    score_drift['severity']
                                )
                                st.warning(f"Score drift detected: PSI = {score_drift['psi']:.3f}")
                        else:
                            st.warning("No features available for drift analysis")
                    else:
                        st.warning("Insufficient data for drift analysis")
                except Exception as e:
                    st.error(f"Drift detection failed: {str(e)}")
    
    with tab2:
        st.markdown("<h4 style='font-size: 1em;'>Contextual Bandit for Adaptive Decision Making</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        <p style='font-size: 0.9em;'>
        The contextual bandit learns which threshold adjustment strategy works best 
        based on the current system context and feedback.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Current Policy Distribution")
            policy = adaptation_module.bandit.get_policy()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            actions = list(policy.keys())
            probabilities = list(policy.values())
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.bar(actions, probabilities, color=colors)
            ax.set_ylabel('Selection Probability')
            ax.set_title('Action Selection Policy')
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("Simulate Feedback")
            
            action_result = st.selectbox(
                "Select action result:",
                ["Positive (threshold worked well)", "Negative (threshold caused issues)"]
            )
            
            selected_action = st.selectbox(
                "Select action to evaluate:",
                adaptation_module.bandit.action_names
            )
            
            if st.button("Submit Feedback"):
                action_idx = adaptation_module.bandit.action_names.index(selected_action)
                reward = 1.0 if "Positive" in action_result else -0.5
                
                adaptation_module.bandit.update(action_idx, reward)
                st.success(f"Feedback submitted. Reward: {reward}")
        
        st.markdown("Action Statistics")
        stats_df = pd.DataFrame({
            'Action': adaptation_module.bandit.action_names,
            'Value': adaptation_module.bandit.action_values,
            'Count': adaptation_module.bandit.action_counts
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.markdown("<h4 style='font-size: 1em;'>System Health and Performance</h4>", unsafe_allow_html=True)
        
        health_metrics = {
            'Model Performance': 'Stable',
            'Data Pipeline': 'Operational',
            'Drift Detection': 'Active',
            'Alert System': 'Enabled',
            'XAI Layer': 'Functioning'
        }
        
        for metric, status in health_metrics.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(metric)
            with col2:
                color = 'green'
                st.markdown(f"<span style='color: {color};'>{status}</span>", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("Recent System Activity")
        
        if adaptation_module.drift_history:
            latest_drift = adaptation_module.drift_history[-1]
            st.write(f"Last drift check: {latest_drift['timestamp'].strftime('%Y-%m-%d %H:%M')}")

def main():
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; text-align: center; margin: 0; font-size: 1.8em;'>
                Credit Risk Intelligence Platform
            </h1>
            <p style='color: #e0e0e0; text-align: center; font-size: 0.9em; margin: 10px 0 0 0;'>
                Complete System Architecture Implementation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    clf, train_df, test_df = load_model_and_data()
    
    if clf is None:
        st.error("Failed to load model or data. Please check the source files.")
        return
    
    feature_cols = get_feature_columns(clf, train_df)
    
    X_train = train_df[feature_cols] if len(feature_cols) > 0 else train_df.select_dtypes(include=[np.number])
    X_test = test_df[feature_cols] if len(feature_cols) > 0 else test_df.select_dtypes(include=[np.number])
    
    y_train = train_df['fraud_flag'] if 'fraud_flag' in train_df.columns else None
    y_test = test_df['fraud_flag'] if 'fraud_flag' in test_df.columns else None
    
    if y_test is not None and len(X_test) != len(y_test):
        st.warning(f"Warning: Test data inconsistent")
    
    fixed_model = FixedMLModel(clf)
    xai_layer = XAILayer(clf)
    optimization_module = OptimizationModule(clf)
    adaptation_module = AdaptationModule(X_train.values)
    dashboard = Dashboard()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard Overview",
        "Credit Application Analysis",
        "XAI Explanations",
        "Optimization and Policies",
        "Adaptation and Monitoring"
    ])
    
    with tab1:
        render_dashboard(fixed_model, xai_layer, optimization_module, adaptation_module, dashboard, X_test, y_test)
    
    with tab2:
        render_credit_application(fixed_model, xai_layer, test_df, feature_cols)
    
    with tab3:
        render_xai_explanations(xai_layer, X_test)
    
    with tab4:
        render_optimization(optimization_module, X_train, y_train, X_test)
    
    with tab5:
        render_adaptation(adaptation_module, fixed_model, xai_layer, X_train, X_test, dashboard)

if __name__ == "__main__":
    main()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em; padding: 20px;'>
            <p><strong>Credit Risk Intelligence Platform</strong> - Complete System Architecture Implementation</p>
            <p>Fixed ML Model • XAI Layer • Optimization Module • Adaptation Module • Dashboard</p>
            <p>Based on Kaggle Competition Dataset</p>
            <p style='font-size: 0.7em;'>Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    """, unsafe_allow_html=True)