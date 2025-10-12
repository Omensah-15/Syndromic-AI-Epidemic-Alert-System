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
            
            data = []
            for i in range(100):
                location = random.choice(locations)
                severity = random.choice(['Mild', 'Moderate', 'Severe', 'Critical'])
                
                # Create comprehensive report data with ALL required fields
                report_data = {
                    'id': i + 1,
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 365)),
                    'location': location,
                    'severity': severity,
                    'cases_count': random.randint(1, 50),
                    'risk_score': random.uniform(0.1, 0.9),
                    'risk_level': random.choice(['Low', 'Medium', 'High', 'Critical']),
                    'syndrome_types': [random.choice(['Gastrointestinal', 'Respiratory', 'Vector-borne'])],
                    'environmental_data': {
                        'temperature': random.uniform(20, 35),
                        'humidity': random.uniform(30, 90),
                        'turbidity': random.uniform(0, 100)
                    }
                }
                
                data.append(report_data)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error loading sample data: {e}")
            # Return empty DataFrame with all required columns
            return pd.DataFrame(columns=[
                'id', 'timestamp', 'location', 'severity', 'cases_count', 
                'risk_score', 'risk_level', 'syndrome_types', 'environmental_data'
            ])
    
    def generate_risk_assessments(self, data):
        assessments = []
        
        if data is None or data.empty:
            return assessments
            
        for _, row in data.iterrows():
            assessment = {
                'location': row.get('location', 'Unknown'),
                'risk_score': row.get('risk_score', 0),
                'risk_level': row.get('risk_level', 'Low'),
                'timestamp': row.get('timestamp', datetime.now()),
                'report_id': row.get('id', 0),
                'syndrome_types': row.get('syndrome_types', []),
                'environmental_data': row.get('environmental_data', {})
            }
            assessments.append(assessment)
        
        return assessments
