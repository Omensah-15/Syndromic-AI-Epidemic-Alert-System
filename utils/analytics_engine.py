import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsEngine:
    """Advanced analytics and anomaly detection"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    def calculate_metrics(self, risk_assessments):
        """Calculate comprehensive analytics metrics"""
        if not risk_assessments:
            return {}
        
        df = pd.DataFrame(risk_assessments)
        
        metrics = {
            'total_reports': len(risk_assessments),
            'high_risk_count': len([r for r in risk_assessments if r.get('risk_level') in ['High', 'Critical']]),
            'avg_risk_score': df['risk_score'].mean() if 'risk_score' in df.columns else 0,
            'outbreak_probability': self._calculate_outbreak_probability(risk_assessments),
            'response_efficiency': random.uniform(0.7, 0.95),  # Simulated metric
            'avg_response_time': random.uniform(2, 48)  # Simulated hours
        }
        
        return metrics
    
    def _calculate_outbreak_probability(self, risk_assessments):
        """Calculate outbreak probability"""
        if not risk_assessments:
            return 0.0
        
        recent_assessments = [r for r in risk_assessments 
                            if (pd.Timestamp.now() - pd.to_datetime(r['timestamp'])).days <= 7]
        
        if not recent_assessments:
            return 0.0
        
        high_risk_ratio = len([r for r in recent_assessments 
                             if r.get('risk_level') in ['High', 'Critical']]) / len(recent_assessments)
        
        avg_recent_risk = sum(r.get('risk_score', 0) for r in recent_assessments) / len(recent_assessments)
        
        return min((high_risk_ratio * 0.6 + avg_recent_risk * 0.4) * 1.2, 1.0)
    
    def detect_anomalies(self, risk_assessments):
        """Detect anomalous reports"""
        if len(risk_assessments) < 10:
            return []
        
        try:
            # Prepare features for anomaly detection
            features = []
            for assessment in risk_assessments:
                if 'environmental_data' in assessment:
                    env_data = assessment['environmental_data']
                    features.append([
                        assessment.get('risk_score', 0),
                        env_data.get('temperature', 25),
                        env_data.get('humidity', 50),
                        env_data.get('turbidity', 0),
                        env_data.get('air_quality', 50)
                    ])
            
            if len(features) < 10:
                return []
            
            # Scale features and detect anomalies
            features_scaled = self.scaler.fit_transform(features)
            anomalies = self.anomaly_detector.fit_predict(features_scaled)
            
            # Return anomalous reports
            anomalous_reports = []
            for i, assessment in enumerate(risk_assessments[:len(anomalies)]):
                if anomalies[i] == -1:  # -1 indicates anomaly
                    anomalous_reports.append({
                        **assessment,
                        'anomaly_score': abs(self.anomaly_detector.score_samples([features_scaled[i]])[0])
                    })
            
            return anomalous_reports
            
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return []
    
    def calculate_trends(self, risk_assessments):
        """Calculate risk trends over time"""
        if not risk_assessments:
            return {}
        
        df = pd.DataFrame(risk_assessments)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate moving average
        df['moving_avg'] = df['risk_score'].rolling(window=7, min_periods=1).mean()
        
        # Calculate trend direction
        if len(df) >= 2:
            recent_trend = np.polyfit(range(len(df)), df['risk_score'], 1)[0]
        else:
            recent_trend = 0
        
        return {
            'trend_direction': 'increasing' if recent_trend > 0.01 else 'decreasing' if recent_trend < -0.01 else 'stable',
            'trend_strength': abs(recent_trend),
            'moving_average': df['moving_avg'].iloc[-1] if not df['moving_avg'].isna().all() else 0
        }
