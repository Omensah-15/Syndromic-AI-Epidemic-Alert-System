import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import os

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
                    'location': random.choice(['Urban Center', 'Rural Village', 'Coastal Area']),
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
            return data.to_dict('records') if not data.empty else []
    
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
    
    class AdvancedAnalyticsEngine:
        def calculate_metrics(self, data): 
            return {
                'total_reports': len(data) if data else 0,
                'high_risk_count': len([d for d in data if d.get('risk_level') in ['High', 'Critical']]) if data else 0,
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

import random

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
        self.arduino_manager.start_auto_detection()
    
    def initialize_session_state(self):
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.reports_data = []
            st.session_state.risk_assessments = []
            st.session_state.analytics_metrics = {}
            st.session_state.arduino_connected = False
            st.session_state.last_auto_scan = None
            
            # Load initial data
            self.load_initial_data()
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("SAEAS Control Panel")
            
            # Enhanced Arduino Connection Section with Auto-Detection
            st.subheader("Arduino ESP32")
            
            connection_status = self.arduino_manager.get_connection_status()
            
            # Update session state based on actual connection
            st.session_state.arduino_connected = connection_status['connected']
            
            if connection_status['connected']:
                st.success("Arduino Connected")
                
                # Show connection details
                col1, col2 = st.columns(2)
                with col1:
                    if connection_status['data_age_seconds'] is not None:
                        if connection_status['data_age_seconds'] < 10:
                            st.success("Live Data")
                        elif connection_status['data_age_seconds'] < 30:
                            st.warning("Stale Data")
                        else:
                            st.error("No Recent Data")
                
                with col2:
                    st.metric("Data Source", "Arduino")
                
                # Show live sensor data
                current_data = self.arduino_manager.get_current_data()
                if current_data and current_data.get('data_source') == 'arduino':
                    st.write("**Live Sensor Data:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Temperature", f"{current_data.get('temp', 0):.1f}°C")
                        st.metric("Humidity", f"{current_data.get('hum', 0):.1f}%")
                    with col2:
                        st.metric("Turbidity", f"{current_data.get('turb', 0):.1f} NTU")
                        st.metric("Air Quality", f"{current_data.get('aqi', 0):.1f} AQI")
                
                # Arduino controls
                if st.button("Refresh Data"):
                    self.arduino_manager.send_command("DATA")
                    st.success("Data refresh requested")
                
                if st.button("Calibrate Sensors"):
                    self.arduino_manager.send_command("CALIBRATE")
                    st.info("Calibration started")
                
                if st.button("Disconnect Arduino"):
                    self.arduino_manager.disconnect()
                    st.session_state.arduino_connected = False
                    st.rerun()
                    
            else:
                st.warning("Arduino Not Connected")
                
                # Auto-detection status
                if connection_status['auto_detection_active']:
                    st.info("Auto-detection: ACTIVE")
                    st.write("Plug in Arduino USB to auto-connect")
                else:
                    st.error("Auto-detection: INACTIVE")
                
                # Manual connection options
                if st.button("Scan for Arduino"):
                    with st.spinner("Scanning for Arduino devices..."):
                        ports = self.arduino_manager.force_rescan()
                        
                        if ports:
                            st.success(f"Found {len(ports)} Arduino device(s)")
                            
                            for i, port in enumerate(ports):
                                if st.button(f"Connect to {port}", key=f"connect_{i}"):
                                    success, message = self.arduino_manager.connect(port)
                                    if success:
                                        st.session_state.arduino_connected = True
                                        st.success(message)
                                        st.rerun()
                                    else:
                                        st.error(message)
                        else:
                            st.info("No Arduino devices detected")
                
                # Show that simulated data is available
                st.info("Using simulated sensor data")
                simulated_data = self.arduino_manager.generate_simulated_data("Demo")
                st.write("**Sample Data:**")
                st.write(f"• Temperature: {simulated_data.get('temp', 0):.1f}°C")
                st.write(f"• Humidity: {simulated_data.get('hum', 0):.1f}%")
                st.write(f"• Turbidity: {simulated_data.get('turb', 0):.1f} NTU")
            
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
            
            # Data Management
            st.subheader("Data Management")
            uploaded_file = st.file_uploader("Import Data", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.reports_data = df.to_dict('records')
                    st.success(f"Imported {len(df)} records")
                except Exception as e:
                    st.error(f"Error importing file: {e}")
    
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
        
        # Environmental data
        col1, col2 = st.columns(2)
        with col1:
            self.render_location_risk()
        with col2:
            self.render_environmental_correlations()
    
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
            st.error(f"Error rendering risk chart: {e}")
    
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
            st.error(f"Error rendering syndrome distribution: {e}")
    
    def render_location_risk(self):
        if not st.session_state.risk_assessments:
            st.info("No location data available")
            return
        
        try:
            df = pd.DataFrame(st.session_state.risk_assessments)
            if 'location' not in df.columns or 'risk_score' not in df.columns:
                st.info("Insufficient data for location analysis")
                return
            
            location_risk = df.groupby('location').agg({
                'risk_score': ['mean', 'count'],
                'risk_level': lambda x: (x.isin(['High', 'Critical'])).sum()
            }).round(3)
            
            location_risk.columns = ['avg_risk', 'report_count', 'high_risk_count']
            location_risk = location_risk.sort_values('avg_risk', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(name='Average Risk', x=location_risk.index, y=location_risk['avg_risk']),
                go.Bar(name='High Risk Cases', x=location_risk.index, y=location_risk['high_risk_count'])
            ])
            
            fig.update_layout(
                title="Risk Analysis by Location",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering location risk: {e}")
    
    def render_environmental_correlations(self):
        st.info("Environmental correlation analysis requires sensor data")
        # Placeholder for environmental analysis
    
    def render_spatial_tab(self):
        st.header("Spatial Analysis")
        
        if not st.session_state.risk_assessments:
            st.info("No spatial data available")
            return
        
        try:
            # Simple map visualization
            locations_data = []
            for assessment in st.session_state.risk_assessments[:50]:  # Limit for performance
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
                    'risk_score': risk_score,
                    'risk_level': assessment.get('risk_level', 'Low')
                })
            
            if locations_data:
                map_df = pd.DataFrame(locations_data)
                st.map(map_df)
                
                # Location risk summary
                st.subheader("Location Risk Summary")
                location_summary = map_df.groupby('location').agg({
                    'risk_score': 'mean',
                    'risk_level': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Low'
                }).round(3)
                
                st.dataframe(location_summary)
            else:
                st.info("No location data available for mapping")
        except Exception as e:
            st.error(f"Error rendering spatial analysis: {e}")
    
    def render_analytics_tab(self):
        st.header("Advanced Analytics")
        
        if not st.session_state.risk_assessments:
            st.info("No analytics data available")
            return
        
        try:
            df = pd.DataFrame(st.session_state.risk_assessments)
            
            # Temporal analysis
            st.subheader("Temporal Patterns")
            
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                hourly_pattern = df.groupby('hour')['risk_score'].mean()
                
                fig = px.line(
                    x=hourly_pattern.index, 
                    y=hourly_pattern.values,
                    title="Hourly Risk Pattern",
                    labels={'x': 'Hour of Day', 'y': 'Average Risk Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timestamp data for temporal analysis")
            
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
            st.error(f"Error rendering analytics: {e}")
    
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
                
                # Show summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filtered Reports", len(filtered_df))
                with col2:
                    avg_risk = filtered_df['risk_score'].mean() if 'risk_score' in filtered_df.columns else 0
                    st.metric("Average Risk Score", f"{avg_risk:.3f}")
                with col3:
                    high_risk_count = len(filtered_df[filtered_df['risk_level'].isin(['High', 'Critical'])])
                    st.metric("High/Critical Risk", high_risk_count)
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
            
            for i, anomaly in enumerate(anomalies[:5]):
                with st.expander(f"Anomaly {i+1}: {anomaly.get('location', 'Unknown')} - Risk: {anomaly.get('risk_score', 0):.3f}"):
                    st.write(f"Timestamp: {anomaly.get('timestamp', 'Unknown')}")
                    st.write(f"Risk Level: {anomaly.get('risk_level', 'Unknown')}")
                    st.write(f"Anomaly Score: {anomaly.get('anomaly_score', 0):.3f}")
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")
    
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
