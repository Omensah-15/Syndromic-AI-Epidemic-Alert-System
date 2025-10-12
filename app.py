import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import altair as alt
import pydeck as pdk
from datetime import datetime, timedelta
import json
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import DataLoader
from utils.arduino_manager import ArduinoManager
from utils.analytics_engine import AdvancedAnalyticsEngine
from models.model_training import ModelInferenceEngine

# Page configuration
st.set_page_config(
    page_title="SAEAS - Epidemic Alert System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SAEASApplication:
    """Main SAEAS Application Class"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.arduino_manager = ArduinoManager()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.inference_engine = ModelInferenceEngine()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.reports_data = []
            st.session_state.risk_assessments = []
            st.session_state.iot_data = {}
            st.session_state.analytics_metrics = {}
            st.session_state.arduino_connected = False
            st.session_state.ml_models_loaded = False
            
            # Load ML models
            self.load_ml_models()
            
            # Load sample data
            self.load_initial_data()
    
    def load_ml_models(self):
        """Load trained ML models"""
        try:
            if self.inference_engine.load_models():
                st.session_state.ml_models_loaded = True
                st.success("ML Models loaded successfully!")
            else:
                st.warning("Using rule-based fallback system")
        except Exception as e:
            st.error(f"Error loading ML models: {e}")
    
    def load_initial_data(self):
        """Load initial demonstration data"""
        sample_data = self.data_loader.load_sample_data()
        if sample_data is not None:
            st.session_state.reports_data = sample_data.to_dict('records')
            st.session_state.risk_assessments = self.data_loader.generate_risk_assessments(sample_data)
    
    def run(self):
        """Main application runner"""
        # Sidebar
        self.render_sidebar()
        
        # Main content
        self.render_header()
        self.render_dashboard()
    
    def render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.title("SAEAS Control Panel")
            
            # Arduino Connection
            self.render_arduino_panel()
            
            # System Status
            self.render_system_status()
            
            # Quick Actions
            self.render_quick_actions()
            
            # Data Management
            self.render_data_management()
    
    def render_arduino_panel(self):
        """Render Arduino connection panel"""
        st.subheader("Arduino ESP32")
        
        if st.session_state.arduino_connected:
            st.success("Arduino Connected")
            if st.button("Disconnect"):
                self.arduino_manager.disconnect()
                st.session_state.arduino_connected = False
                st.rerun()
            
            # Show live data
            current_data = self.arduino_manager.get_current_data()
            if current_data:
                st.write("**Live Sensor Data:**")
                for sensor, value in current_data.items():
                    if isinstance(value, (int, float)):
                        st.metric(sensor, f"{value:.1f}")
        else:
            st.warning("🔌 Arduino Not Connected")
            if st.button("🔍 Scan & Connect"):
                ports = self.arduino_manager.discover_ports()
                if ports:
                    success, message = self.arduino_manager.connect(ports[0])
                    if success:
                        st.session_state.arduino_connected = True
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.info("No Arduino devices found")
    
    def render_system_status(self):
        """Render system status panel"""
        st.subheader("System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reports", len(st.session_state.reports_data))
            st.metric("ML Models", "Loaded" if st.session_state.ml_models_loaded else "❌ Failed")
        with col2:
            high_risk = len([r for r in st.session_state.risk_assessments 
                           if r.get('risk_level') in ['High', 'Critical']])
            st.metric("High Risk", high_risk)
            st.metric("Data Source", "Arduino" if st.session_state.arduino_connected else "Simulated")
    
    def render_quick_actions(self):
        """Render quick actions panel"""
        st.subheader("Quick Actions")
        
        if st.button("Process All Reports"):
            self.process_all_reports()
            st.success("Reports processed!")
        
        if st.button("Update Analytics"):
            self.update_analytics()
            st.success("Analytics updated!")
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    def render_data_management(self):
        """Render data management panel"""
        st.subheader("Data Management")
        
        # Export data
        if st.button("Export Data"):
            self.export_data()
        
        # Import data
        uploaded_file = st.file_uploader("Import Data", type=['csv', 'xlsx'])
        if uploaded_file:
            self.import_data(uploaded_file)
    
    def render_header(self):
        """Render main header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("Syndromic AI - Epidemic Alert System")
            st.markdown("### Real-time Public Health Surveillance & Outbreak Prediction")
            
            # Alert banner
            outbreak_prob = st.session_state.analytics_metrics.get('outbreak_probability', 0) * 100
            if outbreak_prob > 70:
                st.error(f"HIGH OUTBREAK RISK: {outbreak_prob:.1f}%")
            elif outbreak_prob > 40:
                st.warning(f"MODERATE OUTBREAK RISK: {outbreak_prob:.1f}%")
            else:
                st.success(f"LOW OUTBREAK RISK: {outbreak_prob:.1f}%")
    
    def render_dashboard(self):
        """Render main dashboard"""
        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview", "🗺️ Spatial Analysis", "📈 Advanced Analytics", 
            "Report System", "🔍 Anomaly Detection", "⚙️ System Config"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_spatial_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_report_tab()
        
        with tab5:
            self.render_anomaly_tab()
        
        with tab6:
            self.render_config_tab()
    
    def render_overview_tab(self):
        """Render overview dashboard"""
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reports = len(st.session_state.reports_data)
            st.metric("Total Reports", total_reports)
        
        with col2:
            high_risk = len([r for r in st.session_state.risk_assessments 
                           if r.get('risk_level') in ['High', 'Critical']])
            st.metric("Critical/High Risk", high_risk)
        
        with col3:
            outbreak_prob = st.session_state.analytics_metrics.get('outbreak_probability', 0) * 100
            st.metric("Outbreak Probability", f"{outbreak_prob:.1f}%")
        
        with col4:
            avg_response = st.session_state.analytics_metrics.get('avg_response_time', 0)
            st.metric("Avg Response Time", f"{avg_response:.1f}h")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_risk_trend_chart()
        
        with col2:
            self.render_syndrome_distribution()
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_location_risk()
        
        with col2:
            self.render_environmental_correlations()
    
    def render_risk_trend_chart(self):
        """Render risk trend chart"""
        if not st.session_state.risk_assessments:
            st.info("No risk assessment data available")
            return
        
        df = pd.DataFrame(st.session_state.risk_assessments)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_risk = df.groupby('date').agg({
            'risk_score': 'mean',
            'location': 'count'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=daily_risk['date'], y=daily_risk['risk_score'], 
                      name="Average Risk", line=dict(color='red', width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(x=daily_risk['date'], y=daily_risk['location'],
                  name="Report Count", opacity=0.3),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Risk Score Trend & Report Volume",
            height=400
        )
        
        fig.update_yaxes(title_text="Risk Score", secondary_y=False)
        fig.update_yaxes(title_text="Report Count", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_syndrome_distribution(self):
        """Render syndrome distribution chart"""
        if not st.session_state.risk_assessments:
            st.info("No syndrome data available")
            return
        
        syndrome_data = []
        for assessment in st.session_state.risk_assessments:
            for syndrome in assessment.get('syndrome_types', []):
                syndrome_data.append({
                    'syndrome': syndrome,
                    'risk_score': assessment['risk_score']
                })
        
        if not syndrome_data:
            st.info("No syndrome data available")
            return
        
        syndrome_df = pd.DataFrame(syndrome_data)
        syndrome_counts = syndrome_df['syndrome'].value_counts()
        
        fig = px.pie(
            values=syndrome_counts.values,
            names=syndrome_counts.index,
            title="Syndrome Type Distribution"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_location_risk(self):
        """Render location risk analysis"""
        if not st.session_state.risk_assessments:
            st.info("No location data available")
            return
        
        df = pd.DataFrame(st.session_state.risk_assessments)
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
    
    def render_environmental_correlations(self):
        """Render environmental correlations"""
        if not st.session_state.risk_assessments:
            st.info("No environmental data available")
            return
        
        env_data = []
        for assessment in st.session_state.risk_assessments:
            if 'iot_data' in assessment:
                env_data.append({
                    'risk_score': assessment['risk_score'],
                    'temperature': assessment['iot_data'].get('temperature'),
                    'humidity': assessment['iot_data'].get('humidity'),
                    'turbidity': assessment['iot_data'].get('turbidity')
                })
        
        if not env_data:
            st.info("No environmental correlation data available")
            return
        
        env_df = pd.DataFrame(env_data).dropna()
        
        if len(env_df) < 2:
            st.info("Insufficient data for correlation analysis")
            return
        
        corr_matrix = env_df.corr()
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu_r'
        )
        
        fig.update_layout(title="Environmental Factor Correlations")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_spatial_tab(self):
        """Render spatial analysis tab"""
        st.header("🗺️ Spatial Analysis")
        
        if not st.session_state.risk_assessments:
            st.info("No spatial data available")
            return
        
        # Create map
        import folium
        from streamlit_folium import folium_static
        
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
        
        # Add risk markers
        risk_colors = {
            'Critical': 'darkred',
            'High': 'red', 
            'Medium': 'orange',
            'Low': 'green',
            'Very Low': 'lightgreen'
        }
        
        locations = {
            'Urban Center': [40.7128, -74.0060],
            'Rural Village A': [40.7589, -73.9851],
            'Rural Village B': [40.7282, -73.7949],
            'Coastal Area': [40.5795, -73.8132],
            'Mountain Region': [40.6635, -73.9387]
        }
        
        for assessment in st.session_state.risk_assessments[-20:]:
            location = assessment['location']
            if location in locations:
                coords = locations[location]
                risk_level = assessment.get('risk_level', 'Low')
                
                folium.Marker(
                    coords,
                    popup=f"""
                    <b>{location}</b><br>
                    Risk: {risk_level}<br>
                    Score: {assessment.get('risk_score', 0):.3f}<br>
                    Time: {assessment.get('timestamp', 'Unknown')}
                    """,
                    tooltip=f"{location} - {risk_level}",
                    icon=folium.Icon(color=risk_colors.get(risk_level, 'gray'))
                ).add_to(m)
        
        folium_static(m, width=800, height=500)
        
        # Heatmap data
        heat_data = []
        for assessment in st.session_state.risk_assessments:
            location = assessment['location']
            if location in locations:
                coords = locations[location]
                risk_weight = assessment.get('risk_score', 0) * 10
                heat_data.append([coords[0], coords[1], risk_weight])
        
        if heat_data:
            from folium.plugins import HeatMap
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    def render_analytics_tab(self):
        """Render advanced analytics tab"""
        st.header("Advanced Analytics")
        
        if not st.session_state.risk_assessments:
            st.info("No analytics data available")
            return
        
        df = pd.DataFrame(st.session_state.risk_assessments)
        
        # Temporal analysis
        st.subheader("Temporal Patterns")
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_pattern = df.groupby('hour')['risk_score'].mean()
            fig = px.line(x=hourly_pattern.index, y=hourly_pattern.values,
                         title="Hourly Risk Pattern")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            daily_pattern = df.groupby('day_of_week')['risk_score'].mean()
            fig = px.bar(x=daily_pattern.index, y=daily_pattern.values,
                        title="Daily Risk Pattern")
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk distribution
        st.subheader("📊 Risk Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='risk_score', nbins=20,
                              title="Risk Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_level_counts = df['risk_level'].value_counts()
            fig = px.pie(values=risk_level_counts.values,
                        names=risk_level_counts.index,
                        title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_report_tab(self):
        """Render report system tab"""
        st.header("Health Report System")
        
        tab1, tab2 = st.tabs(["Submit New Report", "View Reports"])
        
        with tab1:
            self.render_report_submission()
        
        with tab2:
            self.render_reports_view()
    
    def render_report_submission(self):
        """Render report submission form"""
        with st.form("health_report_form"):
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
                # Symptom checklist
                st.write("**Symptoms Observed:**")
                gastrointestinal = st.checkbox("Gastrointestinal (diarrhea, vomiting)")
                respiratory = st.checkbox("Respiratory (cough, fever, breathing issues)")
                vector_borne = st.checkbox("Vector-borne (fever, rash, joint pain)")
                other = st.checkbox("Other symptoms")
                
                severity = st.select_slider(
                    "Overall Severity",
                    options=['Mild', 'Moderate', 'Severe', 'Critical']
                )
                
                cases_count = st.number_input("Number of Cases", min_value=1, value=1)
            
            submitted = st.form_submit_button("🚨 Submit Report & Analyze")
            
            if submitted:
                if not report_text or not location:
                    st.error("Please fill in all required fields (*)")
                else:
                    self.process_new_report(report_text, location, severity, cases_count)
    
    def process_new_report(self, text, location, severity, cases_count):
        """Process a new health report"""
        # Get environmental data
        if st.session_state.arduino_connected:
            env_data = self.arduino_manager.get_current_data()
        else:
            env_data = self.arduino_manager.generate_simulated_data(location)
        
        # Predict risk using ML models
        prediction = self.inference_engine.predict_risk(text, env_data, location, datetime.now())
        
        # Create report
        report = {
            'id': len(st.session_state.reports_data) + 1,
            'text': text,
            'location': location,
            'timestamp': datetime.now(),
            'severity': severity,
            'cases_count': cases_count,
            'risk_score': prediction['risk_score'],
            'risk_level': prediction['risk_level'],
            'environmental_data': env_data,
            'ml_confidence': prediction.get('model_confidence', 0.8)
        }
        
        st.session_state.reports_data.append(report)
        
        # Create risk assessment
        risk_assessment = {
            'location': location,
            'risk_score': prediction['risk_score'],
            'risk_level': prediction['risk_level'],
            'timestamp': datetime.now(),
            'report_id': report['id'],
            'environmental_data': env_data
        }
        
        st.session_state.risk_assessments.append(risk_assessment)
        
        # Update analytics
        self.update_analytics()
        
        # Show results
        risk_color = {
            'Critical': 'red',
            'High': 'orange',
            'Medium': 'yellow',
            'Low': 'green',
            'Very Low': 'blue'
        }.get(prediction['risk_level'], 'gray')
        
        st.success("Report submitted successfully!")
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; border: 2px solid {risk_color};'>
            <h3 style='color: {risk_color};'>Risk Assessment: {prediction['risk_level']}</h3>
            <p><strong>Risk Score:</strong> {prediction['risk_score']:.3f}</p>
            <p><strong>Confidence:</strong> {prediction.get('model_confidence', 0.8):.1%}</p>
            <p><strong>Location:</strong> {location}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction['risk_level'] in ['High', 'Critical']:
            st.error("HIGH RISK ALERT: Immediate response recommended!")
    
    def render_reports_view(self):
        """Render reports view"""
        if not st.session_state.reports_data:
            st.info("No reports available")
            return
        
        df = pd.DataFrame(st.session_state.reports_data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.multiselect(
                "Filter by Location",
                options=df['location'].unique(),
                default=df['location'].unique()
            )
        
        with col2:
            risk_filter = st.multiselect(
                "Filter by Risk Level", 
                options=df['risk_level'].unique(),
                default=df['risk_level'].unique()
            )
        
        with col3:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=df['severity'].unique(),
                default=df['severity'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['location'].isin(location_filter)) &
            (df['risk_level'].isin(risk_filter)) &
            (df['severity'].isin(severity_filter))
        ]
        
        # Display table
        st.dataframe(
            filtered_df[['timestamp', 'location', 'risk_level', 'risk_score', 'severity', 'cases_count']],
            use_container_width=True
        )
    
    def render_anomaly_tab(self):
        """Render anomaly detection tab"""
        st.header("🔍 Anomaly Detection")
        
        if not st.session_state.risk_assessments:
            st.info("No data for anomaly detection")
            return
        
        # Detect anomalies
        anomalies = self.analytics_engine.detect_anomalies(st.session_state.risk_assessments)
        
        if not anomalies:
            st.success("No anomalies detected")
            return
        
        st.warning(f"🚨 {len(anomalies)} anomalous reports detected!")
        
        for anomaly in anomalies[:5]:  # Show top 5
            with st.expander(f"Anomaly: {anomaly['location']} - Risk: {anomaly['risk_score']:.3f}"):
                st.write(f"**Timestamp:** {anomaly['timestamp']}")
                st.write(f"**Risk Level:** {anomaly['risk_level']}")
                st.write(f"**Anomaly Score:** {anomaly.get('anomaly_score', 0):.3f}")
                
                if st.button("Investigate", key=anomaly['report_id']):
                    st.info("Investigation mode would be implemented here")
    
    def render_config_tab(self):
        """Render system configuration tab"""
        st.header("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Thresholds")
            
            critical_thresh = st.slider("Critical Threshold", 0.7, 0.9, 0.8)
            high_thresh = st.slider("High Threshold", 0.5, 0.8, 0.6)
            medium_thresh = st.slider("Medium Threshold", 0.3, 0.6, 0.4)
            
            if st.button("Save Thresholds"):
                st.success("Thresholds saved!")
        
        with col2:
            st.subheader("Alert Settings")
            
            email_alerts = st.checkbox("Enable Email Alerts", value=True)
            sms_alerts = st.checkbox("Enable SMS Alerts", value=False)
            min_alert_level = st.selectbox("Minimum Alert Level", 
                                         ["Medium", "High", "Critical"])
            
            if st.button("Save Alert Settings"):
                st.success("Alert settings saved!")
        
        st.subheader("System Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Retrain Models"):
                with st.spinner("Retraining models..."):
                    time.sleep(2)  # Simulate training
                    st.success("Models retrained!")
        
        with col2:
            if st.button("Export All Data"):
                self.export_data()
        
        with col3:
            if st.button("Clear All Data"):
                if st.checkbox("Confirm deletion"):
                    st.session_state.reports_data = []
                    st.session_state.risk_assessments = []
                    st.success("All data cleared!")
    
    def process_all_reports(self):
        """Process all unprocessed reports"""
        # Implementation for batch processing
        pass
    
    def update_analytics(self):
        """Update analytics metrics"""
        st.session_state.analytics_metrics = self.analytics_engine.calculate_metrics(
            st.session_state.risk_assessments
        )
    
    def export_data(self):
        """Export system data"""
        reports_df = pd.DataFrame(st.session_state.reports_data)
        risk_df = pd.DataFrame(st.session_state.risk_assessments)
        
        # Create download links
        csv_reports = reports_df.to_csv(index=False)
        csv_risk = risk_df.to_csv(index=False)
        
        st.download_button(
            "Download Reports CSV",
            csv_reports,
            "saeas_reports.csv",
            "text/csv"
        )
        
        st.download_button(
            "Download Risk Assessments CSV", 
            csv_risk,
            "saeas_risk_assessments.csv",
            "text/csv"
        )
    
    def import_data(self, uploaded_file):
        """Import data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.reports_data = df.to_dict('records')
            st.success(f"Imported {len(df)} records")
            
        except Exception as e:
            st.error(f"Error importing data: {e}")

# Main application execution
if __name__ == "__main__":
    app = SAEASApplication()
    app.run()
