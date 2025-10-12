import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataLoader:
    def __init__(self):
        self.sample_data = None
    
    def load_sample_data(self):
        try:
            locations = ['Urban Center', 'Rural Village A', 'Rural Village B', 
                        'Coastal Area', 'Mountain Region']
            
            syndromes = ['Gastrointestinal', 'Respiratory', 'Vector-borne', 'Water-borne']
            risk_levels = ['Very Low', 'Low', 'Medium', 'High', 'Critical']
            severities = ['Mild', 'Moderate', 'Severe', 'Critical']
            
            data = []
            for i in range(100):
                location = random.choice(locations)
                syndrome = random.choice(syndromes)
                severity = random.choice(severities)
                
                # Risk factors based on location and syndrome
                base_risk = random.uniform(0.1, 0.9)
                if location == 'Coastal Area' and syndrome == 'Water-borne':
                    base_risk += 0.3
                elif location == 'Urban Center' and syndrome == 'Respiratory':
                    base_risk += 0.2
                
                risk_score = min(base_risk, 1.0)
                
                # Determine risk level based on score
                if risk_score >= 0.7:
                    risk_level = 'Critical'
                elif risk_score >= 0.5:
                    risk_level = 'High'
                elif risk_score >= 0.3:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                data.append({
                    'id': i + 1,
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 365)),
                    'location': location,
                    'syndrome': syndrome,
                    'syndrome_types': [syndrome],
                    'severity': severity,
                    'cases_count': random.randint(1, 50),
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'environmental_data': {
                        'temperature': random.uniform(20, 35),
                        'humidity': random.uniform(30, 90),
                        'turbidity': random.uniform(0, 100)
                    }
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error loading sample data: {e}")
            return pd.DataFrame()
    
    def generate_risk_assessments(self, data):
        assessments = []
        
        if data is None or data.empty:
            return assessments
            
        for _, row in data.iterrows():
            assessments.append({
                'location': row.get('location', 'Unknown'),
                'risk_score': row.get('risk_score', 0),
                'risk_level': row.get('risk_level', 'Low'),
                'timestamp': row.get('timestamp', datetime.now()),
                'report_id': row.get('id', 0),
                'syndrome_types': row.get('syndrome_types', []),
                'environmental_data': row.get('environmental_data', {})
            })
        
        return assessments
