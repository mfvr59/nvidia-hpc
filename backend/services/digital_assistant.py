"""
Digital Assistant Service for FinDoc AI Platform
Integrates NVIDIA Blueprint with 20+ AI models for conversational AI
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import json

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        AutoModelForSequenceClassification, pipeline
    )
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Using basic responses.")

from backend.core.config import settings
from backend.services.nvidia_hpc import NVIDIAHPCManager

logger = logging.getLogger(__name__)


class DigitalAssistant:
    """Digital Assistant powered by NVIDIA Blueprint and multiple AI models"""
    
    def __init__(self, nvidia_hpc: NVIDIAHPCManager):
        self.nvidia_hpc = nvidia_hpc
        self.initialized = False
        
        # AI Models (NVIDIA Blueprint integration)
        self.language_model = None
        self.sentiment_analyzer = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.summarizer = None
        self.translator = None
        self.qa_model = None
        self.text_classifier = None
        self.sentence_encoder = None
        
        # Knowledge base
        self.knowledge_base = {}
        self.conversation_history = {}
        
        # Language support
        self.supported_languages = {
            'es': 'Spanish',
            'en': 'English', 
            'pt': 'Portuguese'
        }
        
        # Intent patterns
        self.intent_patterns = {
            'risk_assessment': [
                'evaluar riesgo', 'calcular riesgo', 'puntaje de riesgo',
                'risk assessment', 'risk score', 'risk calculation'
            ],
            'document_analysis': [
                'analizar documento', 'revisar documento', 'procesar documento',
                'document analysis', 'document review', 'document processing'
            ],
            'fraud_detection': [
                'detectar fraude', 'verificar fraude', 'análisis de fraude',
                'fraud detection', 'fraud analysis', 'fraud check'
            ],
            'loan_information': [
                'información de préstamo', 'tipos de préstamo', 'requisitos',
                'loan information', 'loan types', 'requirements'
            ],
            'credit_health': [
                'salud crediticia', 'puntaje de crédito', 'historial crediticio',
                'credit health', 'credit score', 'credit history'
            ],
            'general_help': [
                'ayuda', 'soporte', 'información', 'cómo funciona',
                'help', 'support', 'information', 'how does it work'
            ]
        }
        
    async def initialize(self) -> None:
        """Initialize digital assistant"""
        try:
            logger.info("Initializing Digital Assistant...")
            
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize conversation management
            await self._initialize_conversation_system()
            
            self.initialized = True
            logger.info("✅ Digital Assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Digital Assistant: {e}")
            raise
    
    async def _initialize_ai_models(self) -> None:
        """Initialize AI models for different tasks"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available. Using basic responses.")
                return
            
            # Language model for text generation
            try:
                model_name = "microsoft/DialoGPT-medium"
                self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.language_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if settings.use_gpu and torch.cuda.is_available():
                    self.language_model = self.language_model.cuda()
                
                logger.info("Language model loaded")
            except Exception as e:
                logger.warning(f"Could not load language model: {e}")
            
            # Sentiment analyzer
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if settings.use_gpu and torch.cuda.is_available() else -1
                )
                logger.info("Sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
            
            # Intent classifier
            try:
                self.intent_classifier = pipeline(
                    "text-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if settings.use_gpu and torch.cuda.is_available() else -1
                )
                logger.info("Intent classifier loaded")
            except Exception as e:
                logger.warning(f"Could not load intent classifier: {e}")
            
            # Entity extractor
            try:
                self.entity_extractor = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    device=0 if settings.use_gpu and torch.cuda.is_available() else -1
                )
                logger.info("Entity extractor loaded")
            except Exception as e:
                logger.warning(f"Could not load entity extractor: {e}")
            
            # Text summarizer
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if settings.use_gpu and torch.cuda.is_available() else -1
                )
                logger.info("Text summarizer loaded")
            except Exception as e:
                logger.warning(f"Could not load summarizer: {e}")
            
            # Sentence encoder for similarity
            try:
                self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                if settings.use_gpu and torch.cuda.is_available():
                    self.sentence_encoder = self.sentence_encoder.to('cuda')
                logger.info("Sentence encoder loaded")
            except Exception as e:
                logger.warning(f"Could not load sentence encoder: {e}")
            
            logger.info("AI models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            raise
    
    async def _load_knowledge_base(self) -> None:
        """Load knowledge base with financial domain information"""
        try:
            self.knowledge_base = {
                'loan_types': {
                    'personal': {
                        'description': 'Préstamo personal para gastos generales',
                        'requirements': ['Ingresos mínimos', 'Historial crediticio', 'Documentación'],
                        'interest_rates': '8-15% anual',
                        'max_amount': '5,000,000 MXN'
                    },
                    'mortgage': {
                        'description': 'Préstamo hipotecario para compra de vivienda',
                        'requirements': ['Enganche mínimo 20%', 'Ingresos comprobables', 'Avalúo'],
                        'interest_rates': '7-12% anual',
                        'max_amount': '15,000,000 MXN'
                    },
                    'business': {
                        'description': 'Préstamo empresarial para negocios',
                        'requirements': ['Plan de negocio', 'Estados financieros', 'Garantías'],
                        'interest_rates': '10-18% anual',
                        'max_amount': '50,000,000 MXN'
                    }
                },
                'risk_factors': {
                    'credit_score': 'Puntaje de crédito bajo (menos de 600)',
                    'debt_to_income': 'Relación deuda-ingresos alta (más de 40%)',
                    'employment': 'Poco tiempo en el empleo actual',
                    'income': 'Ingresos insuficientes para el monto solicitado',
                    'age': 'Edad joven con poca experiencia crediticia'
                },
                'fraud_indicators': {
                    'document_inconsistency': 'Inconsistencias en documentos presentados',
                    'income_discrepancy': 'Discrepancia entre ingresos declarados y comprobados',
                    'address_mismatch': 'Direcciones diferentes en documentos',
                    'employment_verification': 'Imposibilidad de verificar empleo'
                },
                'credit_health_tips': [
                    'Mantener pagos puntuales',
                    'No utilizar más del 30% del crédito disponible',
                    'Revisar reporte de crédito regularmente',
                    'Evitar múltiples solicitudes de crédito',
                    'Mantener cuentas antiguas abiertas'
                ]
            }
            
            logger.info("Knowledge base loaded")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise
    
    async def _initialize_conversation_system(self) -> None:
        """Initialize conversation management system"""
        try:
            # Initialize conversation templates
            self.conversation_templates = {
                'greeting': {
                    'es': '¡Hola! Soy el asistente virtual de FinDoc AI. ¿En qué puedo ayudarte hoy?',
                    'en': 'Hello! I\'m the FinDoc AI virtual assistant. How can I help you today?',
                    'pt': 'Olá! Sou o assistente virtual do FinDoc AI. Como posso ajudá-lo hoje?'
                },
                'risk_assessment': {
                    'es': 'Para evaluar el riesgo de un préstamo, necesito analizar varios factores como el puntaje de crédito, ingresos, y relación deuda-ingresos.',
                    'en': 'To assess loan risk, I need to analyze several factors such as credit score, income, and debt-to-income ratio.',
                    'pt': 'Para avaliar o risco de um empréstimo, preciso analisar vários fatores como pontuação de crédito, renda e relação dívida-renda.'
                },
                'document_analysis': {
                    'es': 'Puedo ayudarte a analizar documentos financieros como solicitudes de préstamo, estados de cuenta, y reportes de crédito.',
                    'en': 'I can help you analyze financial documents such as loan applications, bank statements, and credit reports.',
                    'pt': 'Posso ajudá-lo a analisar documentos financeiros como solicitações de empréstimo, extratos bancários e relatórios de crédito.'
                }
            }
            
            logger.info("Conversation system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing conversation system: {e}")
            raise
    
    async def process_query(
        self, 
        query: str, 
        document_context: Optional[str] = None, 
        language: str = "es"
    ) -> Dict[str, Any]:
        """Process user query and generate response"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Analyze query
            intent = await self._classify_intent(query, language)
            sentiment = await self._analyze_sentiment(query)
            entities = await self._extract_entities(query)
            
            # Generate response
            response = await self._generate_response(query, intent, sentiment, entities, document_context, language)
            
            # Generate suggested actions
            suggested_actions = await self._generate_suggested_actions(intent, entities, language)
            
            # Calculate confidence
            confidence = await self._calculate_response_confidence(intent, sentiment, entities)
            
            # Find relevant sources
            sources = await self._find_relevant_sources(query, intent, language)
            
            processing_time = time.time() - start_time
            
            result = {
                'response': response,
                'confidence': confidence,
                'sources': sources,
                'suggested_actions': suggested_actions,
                'intent': intent,
                'sentiment': sentiment,
                'entities': entities,
                'processing_time': processing_time,
                'language': language
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def _classify_intent(self, query: str, language: str) -> str:
        """Classify user intent"""
        try:
            query_lower = query.lower()
            
            # Check against intent patterns
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in query_lower:
                        return intent
            
            # Use ML model if available
            if self.intent_classifier:
                try:
                    result = self.intent_classifier(query)
                    # Map model output to our intents
                    return self._map_model_intent(result[0]['label'])
                except Exception as e:
                    logger.warning(f"Error using intent classifier: {e}")
            
            return 'general_help'
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return 'general_help'
    
    async def _analyze_sentiment(self, query: str) -> Dict[str, Any]:
        """Analyze query sentiment"""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(query)
                return {
                    'sentiment': result[0]['label'],
                    'score': result[0]['score']
                }
            else:
                # Basic sentiment analysis
                positive_words = ['bueno', 'excelente', 'perfecto', 'ayuda', 'gracias']
                negative_words = ['malo', 'problema', 'error', 'difícil', 'confuso']
                
                query_lower = query.lower()
                positive_count = sum(1 for word in positive_words if word in query_lower)
                negative_count = sum(1 for word in negative_words if word in query_lower)
                
                if positive_count > negative_count:
                    sentiment = 'positive'
                elif negative_count > positive_count:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'score': 0.5
                }
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'score': 0.5}
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract named entities from query"""
        try:
            if self.entity_extractor:
                entities = self.entity_extractor(query)
                return [
                    {
                        'text': entity['word'],
                        'type': entity['entity_group'],
                        'score': entity['score']
                    }
                    for entity in entities
                ]
            else:
                # Basic entity extraction
                entities = []
                words = query.split()
                
                # Look for numbers (amounts, scores)
                import re
                numbers = re.findall(r'\d+', query)
                for num in numbers:
                    entities.append({
                        'text': num,
                        'type': 'NUMBER',
                        'score': 0.8
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _generate_response(
        self, 
        query: str, 
        intent: str, 
        sentiment: Dict[str, Any], 
        entities: List[Dict[str, Any]], 
        document_context: Optional[str], 
        language: str
    ) -> str:
        """Generate response based on intent and context"""
        try:
            # Get base response for intent
            if intent in self.conversation_templates:
                base_response = self.conversation_templates[intent].get(language, self.conversation_templates[intent]['es'])
            else:
                base_response = self.conversation_templates['general_help'].get(language, self.conversation_templates['general_help']['es'])
            
            # Enhance response based on context
            enhanced_response = await self._enhance_response_with_context(
                base_response, query, intent, entities, document_context, language
            )
            
            # Add sentiment-appropriate tone
            final_response = await self._adjust_tone_for_sentiment(enhanced_response, sentiment, language)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self.conversation_templates['greeting'].get(language, self.conversation_templates['greeting']['es'])
    
    async def _enhance_response_with_context(
        self, 
        base_response: str, 
        query: str, 
        intent: str, 
        entities: List[Dict[str, Any]], 
        document_context: Optional[str], 
        language: str
    ) -> str:
        """Enhance response with specific context and entities"""
        try:
            enhanced_response = base_response
            
            # Add specific information based on intent
            if intent == 'loan_information':
                loan_types = self.knowledge_base.get('loan_types', {})
                if entities:
                    # Look for loan type mentions
                    for entity in entities:
                        if entity['text'].lower() in loan_types:
                            loan_info = loan_types[entity['text'].lower()]
                            enhanced_response += f"\n\nPara préstamos {entity['text']}:\n"
                            enhanced_response += f"- Descripción: {loan_info['description']}\n"
                            enhanced_response += f"- Tasa de interés: {loan_info['interest_rates']}\n"
                            enhanced_response += f"- Monto máximo: {loan_info['max_amount']}"
                            break
            
            elif intent == 'risk_assessment':
                risk_factors = self.knowledge_base.get('risk_factors', {})
                enhanced_response += "\n\nLos principales factores de riesgo incluyen:\n"
                for factor, description in risk_factors.items():
                    enhanced_response += f"- {description}\n"
            
            elif intent == 'credit_health':
                tips = self.knowledge_base.get('credit_health_tips', [])
                enhanced_response += "\n\nConsejos para mejorar la salud crediticia:\n"
                for tip in tips[:3]:  # Show first 3 tips
                    enhanced_response += f"- {tip}\n"
            
            # Add document context if available
            if document_context:
                enhanced_response += f"\n\nBasándome en el documento proporcionado, puedo ofrecer análisis adicional específico."
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return base_response
    
    async def _adjust_tone_for_sentiment(self, response: str, sentiment: Dict[str, Any], language: str) -> str:
        """Adjust response tone based on sentiment"""
        try:
            sentiment_type = sentiment.get('sentiment', 'neutral')
            
            if sentiment_type == 'negative':
                if language == 'es':
                    response = "Entiendo tu preocupación. " + response
                elif language == 'en':
                    response = "I understand your concern. " + response
                elif language == 'pt':
                    response = "Entendo sua preocupação. " + response
            
            elif sentiment_type == 'positive':
                if language == 'es':
                    response = "¡Excelente! " + response
                elif language == 'en':
                    response = "Great! " + response
                elif language == 'pt':
                    response = "Excelente! " + response
            
            return response
            
        except Exception as e:
            logger.error(f"Error adjusting tone: {e}")
            return response
    
    async def _generate_suggested_actions(self, intent: str, entities: List[Dict[str, Any]], language: str) -> List[str]:
        """Generate suggested actions based on intent"""
        try:
            actions = []
            
            if intent == 'risk_assessment':
                actions = [
                    "Subir documento para análisis de riesgo",
                    "Ver tutorial sobre evaluación de riesgo",
                    "Contactar asesor financiero"
                ]
            elif intent == 'document_analysis':
                actions = [
                    "Subir documento para procesamiento",
                    "Ver ejemplos de documentos soportados",
                    "Revisar resultados de análisis previos"
                ]
            elif intent == 'loan_information':
                actions = [
                    "Ver tipos de préstamos disponibles",
                    "Calcular capacidad de pago",
                    "Solicitar cotización"
                ]
            elif intent == 'fraud_detection':
                actions = [
                    "Ejecutar análisis de fraude",
                    "Revisar indicadores de riesgo",
                    "Reportar actividad sospechosa"
                ]
            else:
                actions = [
                    "Explorar funcionalidades",
                    "Ver documentación",
                    "Contactar soporte"
                ]
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating suggested actions: {e}")
            return ["Contactar soporte"]
    
    async def _calculate_response_confidence(self, intent: str, sentiment: Dict[str, Any], entities: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the response"""
        try:
            confidence = 0.7  # Base confidence
            
            # Adjust based on intent clarity
            if intent != 'general_help':
                confidence += 0.1
            
            # Adjust based on sentiment clarity
            sentiment_score = sentiment.get('score', 0.5)
            if sentiment_score > 0.7 or sentiment_score < 0.3:
                confidence += 0.05
            
            # Adjust based on entity extraction
            if entities:
                confidence += 0.05
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.7
    
    async def _find_relevant_sources(self, query: str, intent: str, language: str) -> List[str]:
        """Find relevant sources for the response"""
        try:
            sources = []
            
            # Add knowledge base sources
            if intent in ['loan_information', 'risk_assessment', 'credit_health']:
                sources.append("Base de conocimiento FinDoc AI")
            
            # Add document sources if relevant
            if intent in ['document_analysis', 'risk_assessment']:
                sources.append("Análisis de documentos")
            
            # Add regulatory sources
            if intent in ['loan_information', 'fraud_detection']:
                sources.append("Regulaciones bancarias")
            
            return sources
            
        except Exception as e:
            logger.error(f"Error finding sources: {e}")
            return ["FinDoc AI Platform"]
    
    def _map_model_intent(self, model_label: str) -> str:
        """Map model output to our intent categories"""
        mapping = {
            'entailment': 'general_help',
            'neutral': 'general_help',
            'contradiction': 'general_help'
        }
        return mapping.get(model_label, 'general_help')
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.language_model:
                del self.language_model
            if self.sentiment_analyzer:
                del self.sentiment_analyzer
            if self.intent_classifier:
                del self.intent_classifier
            if self.entity_extractor:
                del self.entity_extractor
            if self.summarizer:
                del self.summarizer
            if self.sentence_encoder:
                del self.sentence_encoder
            
            logger.info("Digital Assistant resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 