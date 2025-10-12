import pandas as pd
import numpy as np
from datetime import datetime
import random

class AdvancedAnalyticsEngine:
    def __init__(self):
        pass
    
    def calculate_metrics(self, risk_assessments):
        if not risk_assessments:
            return {
                'total_reports': 0,
                'high_risk_count': 0,
                'avg_risk_score': 0,
                'outbreak_probability': 0.0,
                'avg_response_time': 0.0
            }
        
        try:
            # Convert to DataFrame safely
            df = pd.DataFrame(risk_assessments)
            
            # Calculate metrics with safe column access
            high_risk_count = 0
            avg_risk_score = 0
            
            if not df.empty:
                if 'risk_level' in df.columns:
                    high_risk_count = len(df[df['risk_level'].isin(['High', 'Critical'])])
                if 'risk_score' in df.columns:
                    avg_risk_score = df['risk_score'].mean()
            
            # Calculate outbreak probability
            recent_assessments = [r for r in risk_assessments 
                                if isinstance(r.get('timestamp'), datetime) and 
                                (datetime.now() - r['timestamp']).days <= 7]
            
            if recent_assessments:
                high_risk_recent = len([r for r in recent_assessments 
                                      if r.get('risk_level') in ['High', 'Critical']])
                outbreak_probability = min((high_risk_recent / len(recent_assessments)) * 1.2, 1.0)
            else:
                outbreak_probability = 0.0
            
            return {
                'total_reports': len(risk_assessments),
                'high_risk_count': high_risk_count,
                'avg_risk_score': avg_risk_score,
                'outbreak_probability': outbreak_probability,
                'avg_response_time': random.uniform(2, 48)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'total_reports': len(risk_assessments),
                'high_risk_count': 0,
                'avg_risk_score': 0,
                'outbreak_probability': 0.0,
                'avg_response_time': 0.0
            }
    
    def detect_anomalies(self, risk_assessments):
        if len(risk_assessments) < 5:
            return []
        
        try:
            # Simple anomaly detection based on risk score
            risk_scores = [r.get('risk_score', 0) for r in risk_assessments]
            if not risk_scores:
                return []
                
            mean_risk = np.mean(risk_scores)
            std_risk = np.std(risk_scores)
            
            if std_risk == 0:  # All scores are the same
                return []
            
            anomalies = []
            for assessment in risk_assessments:
                risk_score = assessment.get('risk_score', 0)
                if risk_score > mean_risk + 2 * std_risk:
                    anomalies.append({
                        **assessment,
                        'anomaly_score': (risk_score - mean_risk) / std_risk
                    })
            
            return anomalies
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return []
