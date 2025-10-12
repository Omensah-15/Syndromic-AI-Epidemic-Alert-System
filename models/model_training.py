# models/model_training.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ModelInferenceEngine:
    """ML model inference engine for risk prediction"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.models_loaded = False
    
    def load_models(self):
        """Load trained models"""
        try:
            model_path = 'models/trained_models/'
            
            # Load text classifier
            if os.path.exists(f'{model_path}text_classifier.pkl'):
                self.models['text_classifier'] = joblib.load(f'{model_path}text_classifier.pkl')
            
            # Load random forest
            if os.path.exists(f'{model_path}random_forest.pkl'):
                self.models['random_forest'] = joblib.load(f'{model_path}random_forest.pkl')
            
            # Load vectorizer
            if os.path.exists(f'{model_path}tfidf_vectorizer.pkl'):
                self.vectorizers['tfidf'] = joblib.load(f'{model_path}tfidf_vectorizer.pkl')
            
            # Load scaler
            if os.path.exists(f'{model_path}scaler.pkl'):
                self.scalers['standard'] = joblib.load(f'{model_path}scaler.pkl')
            
            # Load encoders
            if os.path.exists(f'{model_path}label_encoder.pkl'):
                self.encoders['risk_level'] = joblib.load(f'{model_path}label_encoder.pkl')
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_risk(self, text, environmental_data, location, timestamp):
        """Predict risk using ML models"""
        try:
            if self.models_loaded:
                return self._ml_prediction(text, environmental_data, location, timestamp)
            else:
                return self._rule_based_prediction(text, environmental_data)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction()
    
    def _ml_prediction(self, text, environmental_data, location, timestamp):
        """ML-based risk prediction"""
        # Prepare features
        features = self._prepare_features(text, environmental_data, location, timestamp)
        
        # Text features
        if 'text_classifier' in self.models and 'tfidf' in self.vectorizers:
            text_features = self.vectorizers['tfidf'].transform([text])
            text_pred = self.models['text_classifier'].predict_proba(text_features)[0]
            text_risk = np.argmax(text_pred) / len(text_pred)  # Normalize to 0-1
        else:
            text_risk = self._calculate_text_risk(text)
        
        # Environmental features
        if 'random_forest' in self.models and 'standard' in self.scalers:
            env_features = self._extract_environmental_features(environmental_data)
            env_features_scaled = self.scalers['standard'].transform([env_features])
            env_risk = self.models['random_forest'].predict_proba(env_features_scaled)[0][1]
        else:
            env_risk = self._calculate_environmental_risk(environmental_data)
        
        # Combined risk score
        combined_risk = (text_risk * 0.6 + env_risk * 0.4)
        
        return {
            'risk_score': float(combined_risk),
            'risk_level': self._classify_risk_level(combined_risk),
            'model_confidence': 0.85,
            'components': {
                'text_risk': text_risk,
                'environmental_risk': env_risk
            }
        }
    
    def _rule_based_prediction(self, text, environmental_data):
        """Rule-based fallback prediction"""
        text_risk = self._calculate_text_risk(text)
        env_risk = self._calculate_environmental_risk(environmental_data)
        
        combined_risk = (text_risk * 0.6 + env_risk * 0.4)
        
        return {
            'risk_score': float(combined_risk),
            'risk_level': self._classify_risk_level(combined_risk),
            'model_confidence': 0.7,
            'components': {
                'text_risk': text_risk,
                'environmental_risk': env_risk
            }
        }
    
    def _fallback_prediction(self):
        """Complete fallback prediction"""
        return {
            'risk_score': 0.3,
            'risk_level': 'Low',
            'model_confidence': 0.5,
            'components': {
                'text_risk': 0.3,
                'environmental_risk': 0.3
            }
        }
    
    def _prepare_features(self, text, environmental_data, location, timestamp):
        """Prepare features for ML models"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'urgency_words': self._count_urgency_words(text),
            'symptom_density': self._calculate_symptom_density(text),
            'location_encoded': hash(location) % 100,  # Simple encoding
            'hour_of_day': timestamp.hour
        }
        
        # Add environmental features
        features.update(environmental_data)
        
        return features
    
    def _extract_environmental_features(self, environmental_data):
        """Extract environmental features for ML"""
        return [
            environmental_data.get('temperature', 25),
            environmental_data.get('humidity', 50),
            environmental_data.get('turbidity', 0),
            environmental_data.get('air_quality', 50),
            environmental_data.get('water_ph', 7.0)
        ]
    
    def _calculate_text_risk(self, text):
        """Calculate risk from text analysis"""
        text_lower = text.lower()
        
        # High-risk keywords
        high_risk_words = ['outbreak', 'epidemic', 'critical', 'emergency', 'severe', 
                          'cholera', 'pneumonia', 'dengue', 'hospitalized']
        
        # Medium-risk keywords  
        medium_risk_words = ['multiple cases', 'spreading', 'fever', 'diarrhea', 
                           'vomiting', 'cough', 'breathing difficulty']
        
        high_risk_count = sum(1 for word in high_risk_words if word in text_lower)
        medium_risk_count = sum(1 for word in medium_risk_words if word in text_lower)
        
        risk_score = (high_risk_count * 0.3 + medium_risk_count * 0.15)
        
        return min(risk_score, 1.0)
    
    def _calculate_environmental_risk(self, environmental_data):
        """Calculate risk from environmental factors"""
        turbidity = environmental_data.get('turbidity', 0) / 100
        temperature_risk = abs(environmental_data.get('temperature', 25) - 25) / 20
        humidity_risk = abs(environmental_data.get('humidity', 50) - 60) / 40
        air_quality_risk = environmental_data.get('air_quality', 50) / 200
        
        env_risk = (turbidity * 0.4 + temperature_risk * 0.2 + 
                   humidity_risk * 0.2 + air_quality_risk * 0.2)
        
        return min(env_risk, 1.0)
    
    def _count_urgency_words(self, text):
        """Count urgency-related words"""
        urgency_words = ['urgent', 'immediate', 'emergency', 'critical', 'severe', 
                        'outbreak', 'epidemic', 'crisis', 'alert']
        return sum(1 for word in urgency_words if word in text.lower())
    
    def _calculate_symptom_density(self, text):
        """Calculate symptom keyword density"""
        symptom_words = ['fever', 'cough', 'diarrhea', 'vomiting', 'pain', 
                        'rash', 'headache', 'breathing', 'dehydration']
        words = text.lower().split()
        if not words:
            return 0
        return sum(1 for word in words if word in symptom_words) / len(words)
    
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
