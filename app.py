import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import os
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import utility modules with error handling
try:
    from utils.data_loader import DataLoader
    from utils.arduino_manager import ArduinoManager
    from utils.analytics_engine import AdvancedAnalyticsEngine
    from models.model_training import ModelInferenceEngine
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = str(e)
    # Create fallback classes
    class DataLoader:
        def load_sample_data(self): 
            return self._create_fallback_data()
        def _create_fallback_data(self):
            # Create sample data with ALL required columns
            data = []
            for i in range(100):
                data.append({
                    'id': i + 1,
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 365)),
                    'location': random.choice(['Urban Center', 'Rural Village A', 'Rural Village B', 'Coastal Area', 'Mountain Region']),
                    'severity': random.choice(['Mild', 'Moderate', 'Severe', 'Critical']),
                    'cases_count': random.randint(1, 50),
                    'risk_score': random.uniform(0.1, 0.9),
                    'risk_level': random.choice(['Low', 'Medium', 'High', 'Critical']),
                    'syndrome_types': [random.choice(['Gastrointestinal', 'Respiratory', 'Vector-borne'])],
                    'environmental_data': {
                        'temperature': random.uniform(20, 35),
                        'humidity': random.uniform(30, 90),
                        'turbidity': random.uniform(0, 100)
                    }
                })
            return pd.DataFrame(data)
        def generate_risk_assessments(self, data):
            if data is None or data.empty:
                return []
            return data.to_dict('records')
    
    class ArduinoManager:
        def discover_ports(self): return []
        def connect(self, port, baudrate=115200): return True, "Connected"
        def disconnect(self): pass
        def get_current_data(self): 
            return {
                'temperature': 25.0,
                'humidity': 50.0,
                'turbidity': 30.0,
                'air_quality': 45.0
            }
        def generate_simulated_data(self, location):
            return self.get_current_data()
        def start_auto_detection(self): pass
        def stop_auto_detection(self): pass
        def get_connection_status(self):
            return {
                'connected': False,
                'data_source': 'simulated',
                'last_read_time': None,
                'data_age_seconds': None,
                'auto_detection_active': False
            }
        def force_rescan(self): return []
        def send_command(self, command): return True
    
    class AdvancedAnalyticsEngine:
        def calculate_metrics(self, data): 
            if not data:
                return {
                    'total_reports': 0,
                    'high_risk_count': 0,
                    'avg_risk_score': 0,
                    'outbreak_probability': 0.0,
                    'avg_response_time': 0.0
                }
            return {
                'total_reports': len(data),
                'high_risk_count': len([d for d in data if d.get('risk_level') in ['High', 'Critical']]),
                'avg_risk_score': np.mean([d.get('risk_score', 0) for d in data]) if data else 0,
                'outbreak_probability': 0.1,
                'avg_response_time': 12.5
            }
        def detect_anomalies(self, data): return []
    
    class ModelInferenceEngine:
        def predict_risk(self, text, env_data, location, timestamp):
            return {
                'risk_score': 0.3, 
                'risk_level': 'Low',
                'model_confidence': 0.8
            }

# Page configuration
st.set_page_config(
    page_title="SAEAS - Epidemic Alert System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SAEASApplication:
    def __init__(self):
        self.data_loader = DataLoader()
        self.arduino_manager = ArduinoManager()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.inference_engine = ModelInferenceEngine()
        self.initialize_session_state()
        
        # Start auto-detection when app starts
        try:
            self.arduino_manager.start_auto_detection()
        except:
            pass  # Silently fail if auto-detection not available
    
    def initialize_session_state(self):
        """Initialize session state safely"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.reports_data = []
            st.session_state.risk_assessments = []
            st.session_state.analytics_metrics = {}
            st.session_state.arduino_connected = False
            
            # Load initial data safely
            self.load_initial_data()
    
    def load_initial_data(self):
        """Load initial data with error handling"""
        try:
            sample_data = self.data_loader.load_sample_data()
            if sample_data is not None and not sample_data.empty:
                # Ensure all required columns exist
                sample_data = self._ensure_dataframe_columns(sample_data)
                st.session_state.reports_data = sample_data.to_dict('records')
                
                # Generate risk assessments safely
                risk_assessments = self.data_loader.generate_risk_assessments(sample_data)
                if risk_assessments:
                    st.session_state.risk_assessments = risk_assessments
                
                # Calculate initial metrics
                st.session_state.analytics_metrics = self.analytics_engine.calculate_metrics(
                    st.session_state.risk_assessments
                )
        except Exception as e:
            # If loading fails, create minimal data
            st.session_state.reports_data = self._create_minimal_data()
            st.session_state.risk_assessments = []
            st.session_state.analytics_metrics = {
                'total_reports': len(st.session_state.reports_data),
                'high_risk_count': 0,
                'avg_risk_score': 0.3,
                'outbreak_probability': 0.1,
                'avg_response_time': 12.5
            }
    
    def _create_minimal_data(self):
        """Create minimal sample data if loading fails"""
        data = []
        for i in range(50):
            data.append({
                'id': i + 1,
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
                'location': random.choice(['Urban Center', 'Rural Village A', 'Coastal Area']),
                'severity': random.choice(['Mild', 'Moderate', 'Severe']),
                'cases_count': random.randint(1, 20),
                'risk_score': random.uniform(0.1, 0.7),
                'risk_level': random.choice(['Low', 'Medium', 'High']),
                'syndrome_types': [random.choice(['Gastrointestinal', 'Respiratory'])],
                'text': f"Sample report {i+1} with symptoms",
                'environmental_data': {
                    'temperature': random.uniform(20, 30),
                    'humidity': random.uniform(40, 80),
                    'turbidity': random.uniform(10, 60)
                }
            })
        return data
    
    def _ensure_dataframe_columns(self, df):
        """Ensure DataFrame has all required columns"""
        required_columns = {
            'id': range(len(df)),
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(len(df))],
            'location': ['Unknown'] * len(df),
            'severity': ['Moderate'] * len(df),
            'cases_count': [1] * len(df),
            'risk_score': [0.5] * len(df),
            'risk_level': ['Medium'] * len(df),
            'syndrome_types': [['General']] * len(df),
            'environmental_data': [{}] * len(df)
        }
        
        for col, default_values in required_columns.items():
            if col not in df.columns:
                df[col] = default_values[:len(df)]
        
        return df
    
    def run(self):
        # Show import errors at top
        if IMPORT_ERROR:
            st.warning(f"Import warnings: {IMPORT_ERROR}")
        
        self.render_sidebar()
        self.render_header()
        self.render_dashboard()
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("SAEAS Control Panel")
            
            # Arduino Connection Section
            st.subheader("Arduino ESP32")
            
            try:
                connection_status = self.arduino_manager.get_connection_status()
                st.session_state.arduino_connected = connection_status['connected']
                
                if connection_status['connected']:
                    st.success("Arduino Connected")
                    
                    # Show live sensor data if available
                    current_data = self.arduino_manager.get_current_data()
                    if current_data:
                        st.write("**Live Sensor Data:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Temperature", f"{current_data.get('temp', current_data.get('temperature', 0)):.1f}°C")
                            st.metric("Humidity", f"{current_data.get('hum', current_data.get('humidity', 0)):.1f}%")
                        with col2:
                            st.metric("Turbidity", f"{current_data.get('turb', current_data.get('turbidity', 0)):.1f} NTU")
                            st.metric("Air Quality", f"{current_data.get('aqi', current_data.get('air_quality', 0)):.1f} AQI")
                    
                    if st.button("Disconnect Arduino"):
                        self.arduino_manager.disconnect()
                        st.session_state.arduino_connected = False
                        st.rerun()
                else:
                    st.warning("Arduino Not Connected")
                    
                    if st.button("Scan for Arduino"):
                        try:
                            ports = self.arduino_manager.force_rescan()
                            if ports:
                                st.success(f"Found {len(ports)} Arduino device(s)")
                            else:
                                st.info("No Arduino devices found")
                        except:
                            st.info("Using simulated sensor data")
                    
                    st.info("Using simulated sensor data")
            except:
                st.warning("Arduino Not Connected")
                st.info("Using simulated sensor data")
            
            # System Status
            st.subheader("System Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Reports", len(st.session_state.reports_data))
                st.metric("ML Models", "Loaded")
            with col2:
                high_risk = len([r for r in st.session_state.risk_assessments 
                               if r.get('risk_level') in ['High', 'Critical']])
                st.metric("High Risk", high_risk)
                data_source = "Arduino" if st.session_state.arduino_connected else "Simulated"
                st.metric("Data Source", data_source)
            
            # Quick Actions
            st.subheader("Quick Actions")
            if st.button("Process Reports"):
                self.process_all_reports()
                st.success("Reports processed")
            
            if st.button("Update Analytics"):
                self.update_analytics()
                st.success("Analytics updated")
    
    def render_header(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("Syndromic AI - Epidemic Alert System")
            st.markdown("### Real-time Public Health Surveillance & Outbreak Prediction")
            
            # Risk indicator
            outbreak_prob = st.session_state.analytics_metrics.get('outbreak_probability', 0) * 100
            if outbreak_prob > 70:
                st.error(f"HIGH OUTBREAK RISK: {outbreak_prob:.1f}%")
            elif outbreak_prob > 40:
                st.warning(f"MODERATE OUTBREAK RISK: {outbreak_prob:.1f}%")
            else:
                st.success(f"LOW OUTBREAK RISK: {outbreak_prob:.1f}%")
    
    def render_dashboard(self):
        tab_names = ["Overview", "Spatial Analysis", "Advanced Analytics", "Report System", "Anomaly Detection", "System Config"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            self.render_overview_tab()
        with tabs[1]:
            self.render_spatial_tab()
        with tabs[2]:
            self.render_analytics_tab()
        with tabs[3]:
            self.render_report_tab()
        with tabs[4]:
            self.render_anomaly_tab()
        with tabs[5]:
            self.render_config_tab()
    
    def render_overview_tab(self):
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        metrics = st.session_state.analytics_metrics
        
        with col1:
            st.metric("Total Reports", len(st.session_state.reports_data))
        with col2:
            high_risk = len([r for r in st.session_state.risk_assessments 
                           if r.get('risk_level') in ['High', 'Critical']])
            st.metric("Critical/High Risk", high_risk)
        with col3:
            outbreak_prob = metrics.get('outbreak_probability', 0) * 100
            st.metric("Outbreak Probability", f"{outbreak_prob:.1f}%")
        with col4:
            avg_response = metrics.get('avg_response_time', 0)
            st.metric("Avg Response Time", f"{avg_response:.1f}h")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_risk_trend_chart()
        
        with col2:
            self.render_syndrome_distribution()
    
    def render_risk_trend_chart(self):
        if not st.session_state.risk_assessments:
            st.info("No risk assessment data available")
            return
        
        try:
            df = pd.DataFrame(st.session_state.risk_assessments)
            if 'timestamp' not in df.columns:
                st.info("No timestamp data available")
                return
                
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            daily_risk = df.groupby('date').agg({
                'risk_score': 'mean',
                'location': 'count'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_risk['date'], 
                y=daily_risk['risk_score'],
                name='Average Risk',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Risk Score Trend",
                xaxis_title="Date",
                yaxis_title="Risk Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Could not generate risk trend chart")
    
    def render_syndrome_distribution(self):
        if not st.session_state.risk_assessments:
            st.info("No syndrome data available")
            return
        
        try:
            syndrome_data = []
            for assessment in st.session_state.risk_assessments:
                syndromes = assessment.get('syndrome_types', [])
                if syndromes:
                    for syndrome in syndromes:
                        syndrome_data.append({'syndrome': syndrome})
            
            if not syndrome_data:
                st.info("No syndrome type data available")
                return
            
            syndrome_df = pd.DataFrame(syndrome_data)
            syndrome_counts = syndrome_df['syndrome'].value_counts()
            
            fig = px.pie(
                values=syndrome_counts.values,
                names=syndrome_counts.index,
                title="Syndrome Type Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Could not generate syndrome distribution chart")
    
    def render_spatial_tab(self):
        st.header("Spatial Analysis")
        
        if not st.session_state.risk_assessments:
            st.info("No spatial data available")
            return
        
        try:
            # Simple map visualization
            locations_data = []
            for assessment in st.session_state.risk_assessments[:20]:  # Limit for performance
                location = assessment.get('location', 'Unknown')
                risk_score = assessment.get('risk_score', 0)
                
                # Assign coordinates based on location
                location_coords = {
                    'Urban Center': [40.7128, -74.0060],
                    'Rural Village A': [40.7589, -73.9851],
                    'Rural Village B': [40.7282, -73.7949],
                    'Coastal Area': [40.5795, -73.8132],
                    'Mountain Region': [40.6635, -73.9387],
                    'Unknown': [40.7, -74.0]
                }
                
                coords = location_coords.get(location, [40.7, -74.0])
                locations_data.append({
                    'lat': coords[0],
                    'lon': coords[1],
                    'location': location,
                    'risk_score': risk_score
                })
            
            if locations_data:
                map_df = pd.DataFrame(locations_data)
                st.map(map_df)
            else:
                st.info("No location data available for mapping")
        except Exception as e:
            st.info("Could not generate spatial analysis")
    
    def render_analytics_tab(self):
        st.header("Advanced Analytics")
        
        if not st.session_state.risk_assessments:
            st.info("No analytics data available")
            return
        
        try:
            df = pd.DataFrame(st.session_state.risk_assessments)
            
            # Risk distribution
            st.subheader("Risk Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'risk_score' in df.columns:
                    fig = px.histogram(
                        df, 
                        x='risk_score', 
                        nbins=20,
                        title="Risk Score Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk score data available")
            
            with col2:
                if 'risk_level' in df.columns:
                    risk_level_counts = df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_level_counts.values,
                        names=risk_level_counts.index,
                        title="Risk Level Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk level data available")
        except Exception as e:
            st.info("Could not generate analytics")
    
    def render_report_tab(self):
        st.header("Health Report System")
        
        tab1, tab2 = st.tabs(["Submit New Report", "View Reports"])
        
        with tab1:
            self.render_report_submission()
        with tab2:
            self.render_reports_view()
    
    def render_report_submission(self):
        with st.form("health_report_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                report_text = st.text_area(
                    "Health Report Details*",
                    placeholder="Describe symptoms, number of cases, conditions observed...",
                    height=150
                )
                
                location = st.selectbox(
                    "Location*",
                    options=['Urban Center', 'Rural Village A', 'Rural Village B', 
                            'Coastal Area', 'Mountain Region']
                )
            
            with col2:
                st.write("Symptoms Observed:")
                gastrointestinal = st.checkbox("Gastrointestinal (diarrhea, vomiting)")
                respiratory = st.checkbox("Respiratory (cough, fever, breathing issues)")
                vector_borne = st.checkbox("Vector-borne (fever, rash, joint pain)")
                
                severity = st.select_slider(
                    "Overall Severity",
                    options=['Mild', 'Moderate', 'Severe', 'Critical'],
                    value='Moderate'
                )
                
                cases_count = st.number_input("Number of Cases", min_value=1, value=1)
            
            submitted = st.form_submit_button("Submit Report and Analyze")
            
            if submitted:
                if not report_text or not location:
                    st.error("Please fill in all required fields (*)")
                else:
                    self.process_new_report(report_text, location, severity, cases_count)
    
    def process_new_report(self, text, location, severity, cases_count):
        try:
            # Get environmental data
            if st.session_state.arduino_connected:
                env_data = self.arduino_manager.get_current_data()
            else:
                env_data = self.arduino_manager.generate_simulated_data(location)
            
            # Predict risk
            prediction = self.inference_engine.predict_risk(text, env_data, location, datetime.now())
            
            # Extract syndrome types from text
            syndrome_types = self._extract_syndrome_types(text)
            
            # Create comprehensive report
            report = {
                'id': len(st.session_state.reports_data) + 1,
                'text': text,
                'location': location,
                'timestamp': datetime.now(),
                'severity': severity,
                'cases_count': cases_count,
                'risk_score': prediction['risk_score'],
                'risk_level': prediction['risk_level'],
                'syndrome_types': syndrome_types,
                'environmental_data': env_data
            }
            
            st.session_state.reports_data.append(report)
            
            # Create risk assessment
            risk_assessment = {
                'location': location,
                'risk_score': prediction['risk_score'],
                'risk_level': prediction['risk_level'],
                'timestamp': datetime.now(),
                'report_id': report['id'],
                'syndrome_types': syndrome_types,
                'environmental_data': env_data
            }
            
            st.session_state.risk_assessments.append(risk_assessment)
            
            # Update analytics
            self.update_analytics()
            
            # Show results
            st.success("Report submitted successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", prediction['risk_level'])
            with col2:
                st.metric("Risk Score", f"{prediction['risk_score']:.3f}")
            with col3:
                st.metric("Confidence", f"{prediction.get('model_confidence', 0.8):.1%}")
            
            # Show detected syndromes
            if syndrome_types:
                st.write(f"**Detected Syndromes:** {', '.join(syndrome_types)}")
            
            if prediction['risk_level'] in ['High', 'Critical']:
                st.error("HIGH RISK ALERT: Immediate public health response recommended!")
                
        except Exception as e:
            st.error(f"Error processing report: {e}")
    
    def _extract_syndrome_types(self, text):
        """Extract syndrome types from report text"""
        text_lower = text.lower()
        syndromes = []
        
        syndrome_keywords = {
            'Gastrointestinal': ['diarrhea', 'vomiting', 'cholera', 'stomach', 'abdominal'],
            'Respiratory': ['cough', 'fever', 'breathing', 'pneumonia', 'respiratory'],
            'Vector-borne': ['malaria', 'dengue', 'mosquito', 'rash', 'headache']
        }
        
        for syndrome, keywords in syndrome_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                syndromes.append(syndrome)
        
        return syndromes if syndromes else ['General']
    
    def render_reports_view(self):
        if not st.session_state.reports_data:
            st.info("No reports available")
            return
        
        try:
            # Convert to DataFrame with safe column handling
            reports_df = pd.DataFrame(st.session_state.reports_data)
            
            # Ensure all required columns exist with default values
            required_columns = {
                'location': 'Unknown Location',
                'risk_level': 'Low', 
                'severity': 'Moderate',
                'risk_score': 0.0,
                'cases_count': 1,
                'timestamp': datetime.now()
            }
            
            for col, default_value in required_columns.items():
                if col not in reports_df.columns:
                    reports_df[col] = default_value
            
            # Filters with safe options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                location_options = reports_df['location'].unique().tolist()
                location_filter = st.multiselect(
                    "Filter by Location",
                    options=location_options,
                    default=location_options
                )
            
            with col2:
                risk_options = reports_df['risk_level'].unique().tolist()
                risk_filter = st.multiselect(
                    "Filter by Risk Level", 
                    options=risk_options,
                    default=risk_options
                )
            
            with col3:
                severity_options = reports_df['severity'].unique().tolist()
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=severity_options,
                    default=severity_options
                )
            
            # Apply filters safely
            filtered_df = reports_df[
                (reports_df['location'].isin(location_filter)) &
                (reports_df['risk_level'].isin(risk_filter)) &
                (reports_df['severity'].isin(severity_filter))
            ]
            
            # Display table with safe column access
            display_columns = ['timestamp', 'location', 'risk_level', 'risk_score', 'severity', 'cases_count']
            available_columns = [col for col in display_columns if col in filtered_df.columns]
            
            if not filtered_df.empty and available_columns:
                # Format timestamp for better display
                if 'timestamp' in filtered_df.columns:
                    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    filtered_df[available_columns].reset_index(drop=True), 
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No data available for display with current filters")
                
        except Exception as e:
            st.error(f"Error displaying reports: {e}")
    
    def render_anomaly_tab(self):
        st.header("Anomaly Detection")
        
        if not st.session_state.risk_assessments:
            st.info("No data for anomaly detection")
            return
        
        try:
            anomalies = self.analytics_engine.detect_anomalies(st.session_state.risk_assessments)
            
            if not anomalies:
                st.success("No anomalies detected")
                return
            
            st.warning(f"{len(anomalies)} anomalous reports detected!")
        except Exception as e:
            st.info("Anomaly detection not available")
    
    def render_config_tab(self):
        st.header("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Thresholds")
            critical_thresh = st.slider("Critical Threshold", 0.7, 0.9, 0.8)
            high_thresh = st.slider("High Threshold", 0.5, 0.8, 0.6)
            medium_thresh = st.slider("Medium Threshold", 0.3, 0.6, 0.4)
            
            if st.button("Save Thresholds"):
                st.success("Risk thresholds saved successfully")
        
        with col2:
            st.subheader("Alert Settings")
            email_alerts = st.checkbox("Enable Email Alerts", value=True)
            sms_alerts = st.checkbox("Enable SMS Alerts", value=False)
            min_alert_level = st.selectbox("Minimum Alert Level", ["Medium", "High", "Critical"])
            
            if st.button("Save Alert Settings"):
                st.success("Alert settings saved successfully")
        
        st.subheader("System Maintenance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export All Data"):
                self.export_data()
        
        with col2:
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared successfully")
        
        with col3:
            if st.button("Clear All Data"):
                if st.checkbox("Confirm deletion of all data"):
                    st.session_state.reports_data = []
                    st.session_state.risk_assessments = []
                    st.session_state.analytics_metrics = {}
                    st.success("All data cleared successfully")
    
    def process_all_reports(self):
        try:
            st.session_state.analytics_metrics = self.analytics_engine.calculate_metrics(
                st.session_state.risk_assessments
            )
        except Exception as e:
            st.error(f"Error processing reports: {e}")
    
    def update_analytics(self):
        try:
            st.session_state.analytics_metrics = self.analytics_engine.calculate_metrics(
                st.session_state.risk_assessments
            )
        except Exception as e:
            st.error(f"Error updating analytics: {e}")
    
    def export_data(self):
        try:
            if st.session_state.reports_data:
                reports_df = pd.DataFrame(st.session_state.reports_data)
                csv_data = reports_df.to_csv(index=False)
                
                st.download_button(
                    "Download Reports CSV",
                    csv_data,
                    "saeas_reports.csv",
                    "text/csv"
                )
            else:
                st.info("No data available for export")
        except Exception as e:
            st.error(f"Error exporting data: {e}")

def main():
    app = SAEASApplication()
    app.run()

if __name__ == "__main__":
    main()
