"""
Predictive Analytics Service for FinDoc AI Platform
Handles risk scoring, credit health analysis, and fraud detection
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Machine learning libraries not available.")

from backend.core.config import settings
from backend.services.nvidia_hpc import NVIDIAHPCManager

logger = logging.getLogger(__name__)


class PredictiveAnalytics:
    """Handles predictive analytics for financial risk assessment"""
    
    def __init__(self, nvidia_hpc: NVIDIAHPCManager):
        self.nvidia_hpc = nvidia_hpc
        self.initialized = False
        self.risk_model = None
        self.fraud_model = None
        self.credit_model = None
        self.scaler = None
        self.label_encoders = {}
        
        # Risk categories
        self.risk_categories = {
            'low': {'min': 0, 'max': 30, 'color': 'green'},
            'medium': {'min': 31, 'max': 60, 'color': 'yellow'},
            'high': {'min': 61, 'max': 80, 'color': 'orange'},
            'very_high': {'min': 81, 'max': 100, 'color': 'red'}
        }
        
    async def initialize(self) -> None:
        """Initialize predictive analytics"""
        try:
            logger.info("Initializing Predictive Analytics...")
            
            # Initialize models
            await self._initialize_models()
            
            # Load pre-trained models or train new ones
            await self._load_or_train_models()
            
            self.initialized = True
            logger.info("✅ Predictive Analytics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Predictive Analytics: {e}")
            raise
    
    async def _initialize_models(self) -> None:
        """Initialize machine learning models"""
        try:
            if not ML_AVAILABLE:
                logger.warning("ML libraries not available. Using rule-based scoring.")
                return
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Initialize label encoders for categorical variables
            categorical_features = ['loan_type', 'employment_status', 'marital_status', 'education_level']
            for feature in categorical_features:
                self.label_encoders[feature] = LabelEncoder()
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def _load_or_train_models(self) -> None:
        """Load pre-trained models or train new ones"""
        try:
            if not ML_AVAILABLE:
                return
            
            # For now, we'll use simple models
            # In production, you'd load pre-trained models from disk
            
            # Risk scoring model (Random Forest)
            self.risk_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Fraud detection model (Gradient Boosting)
            self.fraud_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Credit health model (Random Forest)
            self.credit_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            
            # Train models with sample data (in production, use real data)
            await self._train_models_with_sample_data()
            
            logger.info("Models trained with sample data")
            
        except Exception as e:
            logger.error(f"Error loading/training models: {e}")
            raise
    
    async def _train_models_with_sample_data(self) -> None:
        """Train models with sample data for demonstration"""
        try:
            # Generate sample data for training
            np.random.seed(42)
            n_samples = 1000
            
            # Sample features
            data = {
                'age': np.random.normal(35, 10, n_samples),
                'income': np.random.normal(50000, 20000, n_samples),
                'loan_amount': np.random.normal(100000, 50000, n_samples),
                'credit_score': np.random.normal(650, 100, n_samples),
                'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
                'employment_years': np.random.normal(5, 3, n_samples),
                'loan_type': np.random.choice(['personal', 'mortgage', 'business'], n_samples),
                'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed'], n_samples),
                'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples),
                'education_level': np.random.choice(['high_school', 'bachelor', 'master'], n_samples)
            }
            
            df = pd.DataFrame(data)
            
            # Create target variables
            # Risk score (0-100)
            risk_score = (
                (100 - df['credit_score']) * 0.3 +
                df['debt_to_income'] * 200 +
                (50 - df['age']) * 0.5 +
                np.random.normal(0, 10, n_samples)
            )
            risk_score = np.clip(risk_score, 0, 100)
            
            # Fraud probability (0-1)
            fraud_prob = (
                (df['income'] < 30000).astype(int) * 0.3 +
                (df['debt_to_income'] > 0.5).astype(int) * 0.4 +
                np.random.normal(0, 0.1, n_samples)
            )
            fraud_prob = np.clip(fraud_prob, 0, 1)
            
            # Credit health (good/bad)
            credit_health = (df['credit_score'] > 650).astype(int)
            
            # Prepare features
            feature_columns = ['age', 'income', 'loan_amount', 'credit_score', 'debt_to_income', 'employment_years']
            X = df[feature_columns].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train risk model
            risk_target = (risk_score > 50).astype(int)  # Binary classification
            self.risk_model.fit(X_scaled, risk_target)
            
            # Train fraud model
            self.fraud_model.fit(X_scaled, fraud_prob)
            
            # Train credit health model
            self.credit_model.fit(X_scaled, credit_health)
            
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    async def calculate_risk_score(
        self, 
        document_id: str, 
        customer_data: Dict[str, Any], 
        loan_amount: float, 
        loan_type: str
    ) -> Dict[str, Any]:
        """Calculate risk score for loan application"""
        start_time = time.time()
        
        try:
            logger.info(f"Calculating risk score for document {document_id}")
            
            # Extract features from customer data
            features = await self._extract_features(customer_data, loan_amount, loan_type)
            
            # Calculate risk score
            if ML_AVAILABLE and self.risk_model is not None:
                risk_score = await self._calculate_ml_risk_score(features)
            else:
                risk_score = await self._calculate_rule_based_risk_score(features)
            
            # Determine risk category
            risk_category = self._categorize_risk(risk_score)
            
            # Generate factors and recommendations
            factors = await self._identify_risk_factors(features, risk_score)
            recommendations = await self._generate_recommendations(features, risk_score)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(features, risk_score)
            
            processing_time = time.time() - start_time
            
            result = {
                'risk_score': risk_score,
                'risk_category': risk_category,
                'confidence': confidence,
                'factors': factors,
                'recommendations': recommendations,
                'processing_time': processing_time,
                'features_used': list(features.keys())
            }
            
            logger.info(f"Risk score calculated: {risk_score} ({risk_category})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            raise
    
    async def _extract_features(self, customer_data: Dict[str, Any], loan_amount: float, loan_type: str) -> Dict[str, float]:
        """Extract features from customer data"""
        try:
            features = {
                'age': float(customer_data.get('age', 35)),
                'income': float(customer_data.get('income', 50000)),
                'loan_amount': float(loan_amount),
                'credit_score': float(customer_data.get('credit_score', 650)),
                'debt_to_income': float(customer_data.get('debt_to_income', 0.3)),
                'employment_years': float(customer_data.get('employment_years', 5)),
                'loan_type': loan_type,
                'employment_status': customer_data.get('employment_status', 'employed'),
                'marital_status': customer_data.get('marital_status', 'single'),
                'education_level': customer_data.get('education_level', 'bachelor')
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    async def _calculate_ml_risk_score(self, features: Dict[str, Any]) -> float:
        """Calculate risk score using machine learning model"""
        try:
            # Prepare feature vector
            feature_columns = ['age', 'income', 'loan_amount', 'credit_score', 'debt_to_income', 'employment_years']
            feature_vector = np.array([[features[col] for col in feature_columns]])
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get prediction probability
            risk_prob = self.risk_model.predict_proba(feature_vector_scaled)[0][1]
            
            # Convert to risk score (0-100)
            risk_score = risk_prob * 100
            
            return float(risk_score)
            
        except Exception as e:
            logger.error(f"Error calculating ML risk score: {e}")
            return 50.0  # Default medium risk
    
    async def _calculate_rule_based_risk_score(self, features: Dict[str, Any]) -> float:
        """Calculate risk score using rule-based approach"""
        try:
            risk_score = 0
            
            # Credit score factor (0-30 points)
            credit_score = features['credit_score']
            if credit_score < 500:
                risk_score += 30
            elif credit_score < 600:
                risk_score += 20
            elif credit_score < 700:
                risk_score += 10
            elif credit_score < 800:
                risk_score += 5
            
            # Debt-to-income ratio factor (0-25 points)
            dti = features['debt_to_income']
            if dti > 0.5:
                risk_score += 25
            elif dti > 0.4:
                risk_score += 20
            elif dti > 0.3:
                risk_score += 15
            elif dti > 0.2:
                risk_score += 10
            
            # Income factor (0-20 points)
            income = features['income']
            loan_amount = features['loan_amount']
            income_ratio = income / loan_amount if loan_amount > 0 else 0
            
            if income_ratio < 0.1:
                risk_score += 20
            elif income_ratio < 0.2:
                risk_score += 15
            elif income_ratio < 0.3:
                risk_score += 10
            elif income_ratio < 0.5:
                risk_score += 5
            
            # Employment factor (0-15 points)
            employment_years = features['employment_years']
            if employment_years < 1:
                risk_score += 15
            elif employment_years < 2:
                risk_score += 10
            elif employment_years < 5:
                risk_score += 5
            
            # Age factor (0-10 points)
            age = features['age']
            if age < 25:
                risk_score += 10
            elif age < 30:
                risk_score += 5
            
            return min(risk_score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating rule-based risk score: {e}")
            return 50.0
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk level"""
        for category, range_info in self.risk_categories.items():
            if range_info['min'] <= risk_score <= range_info['max']:
                return category
        return 'very_high'
    
    async def _identify_risk_factors(self, features: Dict[str, Any], risk_score: float) -> List[str]:
        """Identify key risk factors"""
        factors = []
        
        try:
            # Credit score factors
            if features['credit_score'] < 600:
                factors.append("Bajo puntaje de crédito")
            
            # Debt-to-income factors
            if features['debt_to_income'] > 0.4:
                factors.append("Alta relación deuda-ingresos")
            
            # Income factors
            income_ratio = features['income'] / features['loan_amount'] if features['loan_amount'] > 0 else 0
            if income_ratio < 0.2:
                factors.append("Ingresos insuficientes para el monto del préstamo")
            
            # Employment factors
            if features['employment_years'] < 2:
                factors.append("Poco tiempo en el empleo actual")
            
            # Age factors
            if features['age'] < 25:
                factors.append("Edad joven - menor experiencia crediticia")
            
            # Loan type factors
            if features['loan_type'] == 'business':
                factors.append("Préstamo empresarial - mayor riesgo")
            
            return factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return ["Error al analizar factores de riesgo"]
    
    async def _generate_recommendations(self, features: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate recommendations based on risk factors"""
        recommendations = []
        
        try:
            if risk_score > 70:
                recommendations.append("Considerar reducir el monto del préstamo")
                recommendations.append("Solicitar garantía adicional")
                recommendations.append("Revisar capacidad de pago del cliente")
            
            if features['credit_score'] < 600:
                recommendations.append("Recomendar mejorar el puntaje de crédito")
                recommendations.append("Considerar productos de crédito con garantía")
            
            if features['debt_to_income'] > 0.4:
                recommendations.append("Sugerir consolidación de deudas")
                recommendations.append("Evaluar capacidad de pago adicional")
            
            if features['employment_years'] < 2:
                recommendations.append("Verificar estabilidad laboral")
                recommendations.append("Solicitar referencias laborales adicionales")
            
            if features['age'] < 25:
                recommendations.append("Considerar co-deudor o garante")
                recommendations.append("Evaluar historial crediticio familiar")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error al generar recomendaciones"]
    
    async def _calculate_confidence(self, features: Dict[str, Any], risk_score: float) -> float:
        """Calculate confidence in the risk assessment"""
        try:
            confidence = 0.8  # Base confidence
            
            # Adjust based on data quality
            if all(features.values()):
                confidence += 0.1
            
            # Adjust based on risk score extremity
            if risk_score < 20 or risk_score > 80:
                confidence += 0.05
            
            # Adjust based on feature completeness
            missing_features = sum(1 for v in features.values() if v is None or v == 0)
            confidence -= missing_features * 0.02
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.7
    
    async def detect_fraud(self, customer_data: Dict[str, Any], document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential fraud in application"""
        try:
            # Extract fraud indicators
            fraud_indicators = await self._extract_fraud_indicators(customer_data, document_data)
            
            # Calculate fraud probability
            if ML_AVAILABLE and self.fraud_model is not None:
                fraud_prob = await self._calculate_ml_fraud_probability(fraud_indicators)
            else:
                fraud_prob = await self._calculate_rule_based_fraud_probability(fraud_indicators)
            
            # Generate fraud alerts
            alerts = await self._generate_fraud_alerts(fraud_indicators, fraud_prob)
            
            return {
                'fraud_probability': fraud_prob,
                'risk_level': 'high' if fraud_prob > 0.7 else 'medium' if fraud_prob > 0.4 else 'low',
                'alerts': alerts,
                'indicators': fraud_indicators
            }
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            raise
    
    async def _extract_fraud_indicators(self, customer_data: Dict[str, Any], document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fraud indicators from data"""
        indicators = {
            'income_discrepancy': 0,
            'document_inconsistency': 0,
            'address_mismatch': 0,
            'employment_verification': 0,
            'credit_history_gaps': 0
        }
        
        # Add logic to detect fraud indicators
        # This is a simplified version
        
        return indicators
    
    async def _calculate_ml_fraud_probability(self, indicators: Dict[str, Any]) -> float:
        """Calculate fraud probability using ML model"""
        try:
            # Convert indicators to feature vector
            feature_vector = np.array([[v for v in indicators.values()]])
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            fraud_prob = self.fraud_model.predict(feature_vector_scaled)[0]
            return float(fraud_prob)
            
        except Exception as e:
            logger.error(f"Error calculating ML fraud probability: {e}")
            return 0.1
    
    async def _calculate_rule_based_fraud_probability(self, indicators: Dict[str, Any]) -> float:
        """Calculate fraud probability using rules"""
        try:
            total_indicators = sum(indicators.values())
            fraud_prob = min(total_indicators * 0.2, 1.0)
            return fraud_prob
            
        except Exception as e:
            logger.error(f"Error calculating rule-based fraud probability: {e}")
            return 0.1
    
    async def _generate_fraud_alerts(self, indicators: Dict[str, Any], fraud_prob: float) -> List[str]:
        """Generate fraud alerts"""
        alerts = []
        
        if fraud_prob > 0.7:
            alerts.append("ALTA PROBABILIDAD DE FRAUDE - Requiere investigación inmediata")
        elif fraud_prob > 0.4:
            alerts.append("PROBABILIDAD MODERADA DE FRAUDE - Verificación adicional recomendada")
        
        for indicator, value in indicators.items():
            if value > 0:
                alerts.append(f"Indicador de fraude detectado: {indicator}")
        
        return alerts
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get analytics dashboard data"""
        try:
            # This would typically fetch data from a database
            # For now, return sample data
            
            dashboard_data = {
                'total_applications': 1250,
                'approved_applications': 890,
                'rejected_applications': 360,
                'average_risk_score': 45.2,
                'fraud_detection_rate': 0.08,
                'processing_time_avg': 2.3,
                'risk_distribution': {
                    'low': 35,
                    'medium': 40,
                    'high': 20,
                    'very_high': 5
                },
                'monthly_trends': {
                    'applications': [120, 135, 142, 128, 156, 145],
                    'approvals': [85, 95, 102, 88, 110, 98],
                    'risk_scores': [42, 45, 48, 43, 47, 44]
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.risk_model:
                del self.risk_model
            if self.fraud_model:
                del self.fraud_model
            if self.credit_model:
                del self.credit_model
            
            logger.info("Predictive Analytics resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 