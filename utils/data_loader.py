import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataLoader:
    """Data loading and management utilities"""
    
    def __init__(self):
        self.sample_data = None
    
    def load_sample_data(self):
        """Load sample public health data"""
        try:
            # Generate realistic sample data
            locations = ['Urban Center', 'Rural Village A', 'Rural Village B', 
                        'Coastal Area', 'Mountain Region']
            
            syndromes = ['Gastrointestinal', 'Respiratory', 'Vector-borne', 'Water-borne']
            
            data = []
            for i in range(100):
                location = random.choice(locations)
                syndrome = random.choice(syndromes)
                
                # Risk factors based on location and syndrome
                base_risk = random.uniform(0.1, 0.9)
                if location == 'Coastal Area' and syndrome == 'Water-borne':
                    base_risk += 0.3
                elif location == 'Urban Center' and syndrome == 'Respiratory':
                    base_risk += 0.2
                
                data.append({
                    'id': i + 1,
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 365)),
                    'location': location,
                    'syndrome': syndrome,
                    'cases_count': random.randint(1, 50),
                    'risk_score': min(base_risk, 1.0),
                    'risk_level': self._classify_risk_level(base_risk),
                    'environmental_data': {
                        'temperature': random.uniform(20, 35),
                        'humidity': random.uniform(30, 90),
                        'turbidity': random.uniform(0, 100)
                    }
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error loading sample data: {e}")
            return None
    
    def _classify_risk_level(self, risk_score):
        """Classify risk level based on score"""
        if risk_score >= 0.7:
            return 'Critical'
        elif risk_score >= 0.5:
            return 'High'
        elif risk_score >= 0.3:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_risk_assessments(self, data):
        """Generate risk assessments from data"""
        assessments = []
        
        for _, row in data.iterrows():
            assessments.append({
                'location': row['location'],
                'risk_score': row['risk_score'],
                'risk_level': row['risk_level'],
                'timestamp': row['timestamp'],
                'report_id': row['id'],
                'environmental_data': row['environmental_data']
            })
        
        return assessments
