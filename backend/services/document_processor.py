"""
Document Processor Service for FinDoc AI Platform
Handles OCR, entity extraction, and document classification
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import io

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import spacy

# Optional imports with fallbacks
try:
    from transformers import LayoutLMv2Processor, LayoutLMv2ForSequenceClassification  # type: ignore
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Using basic OCR only.")

try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Using Tesseract OCR only.")

from backend.core.config import settings
from backend.services.nvidia_hpc import NVIDIAHPCManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing with OCR and AI-powered extraction"""
    
    def __init__(self, nvidia_hpc: NVIDIAHPCManager):
        self.nvidia_hpc = nvidia_hpc
        self.initialized = False
        self.ocr_engine = None
        self.layout_model = None
        self.nlp_model = None
        self.document_types = {
            'loan_application': 'Solicitud de préstamo',
            'credit_report': 'Reporte de crédito',
            'bank_statement': 'Estado de cuenta',
            'invoice': 'Factura',
            'contract': 'Contrato',
            'identity_document': 'Documento de identidad',
            'income_statement': 'Estado de resultados',
            'balance_sheet': 'Balance general'
        }
        
    async def initialize(self) -> None:
        """Initialize document processor"""
        try:
            logger.info("Initializing Document Processor...")
            
            # Initialize OCR engine
            await self._initialize_ocr()
            
            # Initialize layout analysis model
            await self._initialize_layout_model()
            
            # Initialize NLP model for Spanish
            await self._initialize_nlp_model()
            
            # Create directories
            Path(settings.upload_dir).mkdir(exist_ok=True)
            Path(settings.processed_dir).mkdir(exist_ok=True)
            
            self.initialized = True
            logger.info("✅ Document Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Processor: {e}")
            raise
    
    async def _initialize_ocr(self) -> None:
        """Initialize OCR engines"""
        try:
            # Initialize PaddleOCR for better accuracy
            if PADDLEOCR_AVAILABLE:
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='es'
                )
                logger.info("PaddleOCR initialized with Spanish language support")
            else:
                # Fallback to Tesseract
                pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
                logger.info("Tesseract OCR initialized")
                
        except Exception as e:
            logger.error(f"Error initializing OCR: {e}")
            raise
    
    async def _initialize_layout_model(self) -> None:
        """Initialize layout analysis model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Load LayoutLM model for document understanding
                model_name = "microsoft/layoutlmv2-base-uncased"
                self.layout_model = LayoutLMv2ForSequenceClassification.from_pretrained(model_name)
                self.layout_processor = LayoutLMv2Processor.from_pretrained(model_name)
                
                if settings.use_gpu and torch.cuda.is_available():
                    self.layout_model = self.layout_model.cuda()
                
                logger.info("LayoutLM model loaded for document layout analysis")
                
        except Exception as e:
            logger.warning(f"Could not load LayoutLM model: {e}")
    
    async def _initialize_nlp_model(self) -> None:
        """Initialize NLP model for Spanish"""
        try:
            # Load Spanish language model
            self.nlp_model = spacy.load("es_core_news_lg")
            logger.info("Spanish NLP model loaded")
            
        except Exception as e:
            logger.warning(f"Could not load Spanish NLP model: {e}")
            # Fallback to basic Spanish model
            try:
                self.nlp_model = spacy.load("es_core_news_sm")
                logger.info("Spanish NLP model (small) loaded")
            except:
                logger.warning("No Spanish NLP model available")
    
    async def process_document(self, file) -> Dict[str, Any]:
        """Process uploaded document"""
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing document {document_id}")
            
            # Save uploaded file
            file_path = Path(settings.upload_dir) / f"{document_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Convert to images if PDF
            images = await self._convert_to_images(file_path)
            
            # Process each page
            extracted_data = []
            total_confidence = 0
            
            for i, image in enumerate(images):
                page_data = await self._process_page(image, i + 1)
                extracted_data.append(page_data)
                total_confidence += page_data.get('confidence', 0)
            
            # Determine document type
            document_type = await self._classify_document(extracted_data)
            
            # Extract entities
            entities = await self._extract_entities(extracted_data)
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(images) if images else 0
            
            processing_time = time.time() - start_time
            
            result = {
                'document_id': document_id,
                'document_type': document_type,
                'extracted_data': {
                    'pages': extracted_data,
                    'entities': entities,
                    'summary': await self._generate_summary(extracted_data)
                },
                'confidence_score': avg_confidence,
                'processing_time': processing_time,
                'file_path': str(file_path)
            }
            
            logger.info(f"Document {document_id} processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            raise
    
    async def _convert_to_images(self, file_path: Path) -> List[Image.Image]:
        """Convert document to images"""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                # Convert PDF to images
                images = convert_from_path(
                    file_path,
                    dpi=300,
                    fmt='PNG'
                )
                return images
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                # Load image directly
                image = Image.open(file_path)
                return [image]
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error converting document to images: {e}")
            raise
    
    async def _process_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Process single page"""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            processed_image = await self._preprocess_image(cv_image)
            
            # Extract text using OCR
            text_data = await self._extract_text(processed_image)
            
            # Analyze layout
            layout_data = await self._analyze_layout(processed_image)
            
            # Extract tables
            tables = await self._extract_tables(processed_image)
            
            return {
                'page_number': page_num,
                'text': text_data['text'],
                'confidence': text_data['confidence'],
                'layout': layout_data,
                'tables': tables,
                'image_size': image.size
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return {
                'page_number': page_num,
                'error': str(e),
                'confidence': 0
            }
    
    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    async def _extract_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            if self.ocr_engine and hasattr(self.ocr_engine, 'ocr'):
                # Use PaddleOCR
                result = self.ocr_engine.ocr(image, cls=True)
                
                text_lines = []
                total_confidence = 0
                line_count = 0
                
                for line in result:
                    if line:
                        for word_info in line:
                            if len(word_info) >= 2:
                                text = word_info[1][0]
                                confidence = word_info[1][1]
                                text_lines.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': word_info[0]
                                })
                                total_confidence += confidence
                                line_count += 1
                
                avg_confidence = total_confidence / line_count if line_count > 0 else 0
                full_text = ' '.join([line['text'] for line in text_lines])
                
            else:
                # Use Tesseract
                text = pytesseract.image_to_string(
                    image, 
                    lang='spa+eng',
                    config='--psm 6'
                )
                
                # Get confidence data
                data = pytesseract.image_to_data(
                    image, 
                    lang='spa+eng',
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                text_lines = [{'text': text.strip(), 'confidence': avg_confidence}]
                full_text = text.strip()
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'lines': text_lines
            }
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return {
                'text': '',
                'confidence': 0,
                'lines': []
            }
    
    async def _analyze_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze document layout"""
        try:
            # Basic layout analysis using OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            # Detect rectangles (potential tables/forms)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangles.append({'x': x, 'y': y, 'width': w, 'height': h, 'area': area})
            
            return {
                'lines_detected': len(lines) if lines is not None else 0,
                'rectangles_detected': len(rectangles),
                'layout_type': self._determine_layout_type(rectangles, lines)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing layout: {e}")
            return {'error': str(e)}
    
    async def _extract_tables(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract tables from image"""
        try:
            # Basic table detection using contour analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Potential table
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if it looks like a table
                    if 0.5 < aspect_ratio < 2.0:
                        table_region = image[y:y+h, x:x+w]
                        table_text = await self._extract_text(table_region)
                        
                        tables.append({
                            'bbox': [x, y, w, h],
                            'text': table_text['text'],
                            'confidence': table_text['confidence']
                        })
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    async def _classify_document(self, extracted_data: List[Dict[str, Any]]) -> str:
        """Classify document type"""
        try:
            # Combine all text from pages
            all_text = ' '.join([page.get('text', '') for page in extracted_data])
            all_text_lower = all_text.lower()
            
            # Simple keyword-based classification
            keywords = {
                'loan_application': ['préstamo', 'solicitud', 'crédito', 'loan', 'application'],
                'credit_report': ['reporte', 'crédito', 'credit', 'report', 'buro'],
                'bank_statement': ['estado', 'cuenta', 'statement', 'balance', 'movimiento'],
                'invoice': ['factura', 'invoice', 'recibo', 'cobro'],
                'contract': ['contrato', 'contract', 'acuerdo', 'agreement'],
                'identity_document': ['identificación', 'pasaporte', 'dni', 'curp'],
                'income_statement': ['ingresos', 'ganancias', 'pérdidas', 'income'],
                'balance_sheet': ['balance', 'activos', 'pasivos', 'patrimonio']
            }
            
            scores = {}
            for doc_type, doc_keywords in keywords.items():
                score = sum(1 for keyword in doc_keywords if keyword in all_text_lower)
                scores[doc_type] = score
            
            # Return document type with highest score
            if scores:
                best_type = max(scores.items(), key=lambda x: x[1])[0]
                return best_type if scores[best_type] > 0 else 'unknown'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return 'unknown'
    
    async def _extract_entities(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract named entities from document"""
        try:
            if not self.nlp_model:
                return {}
            
            all_text = ' '.join([page.get('text', '') for page in extracted_data])
            
            # Process text with spaCy
            doc = self.nlp_model(all_text)
            
            entities = {
                'persons': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'money': [],
                'numbers': []
            }
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'LOC':
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                elif ent.label_ == 'MONEY':
                    entities['money'].append(ent.text)
                elif ent.label_ == 'CARDINAL':
                    entities['numbers'].append(ent.text)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    async def _generate_summary(self, extracted_data: List[Dict[str, Any]]) -> str:
        """Generate summary of extracted data"""
        try:
            # Simple summary generation
            total_pages = len(extracted_data)
            total_text_length = sum(len(page.get('text', '')) for page in extracted_data)
            avg_confidence = sum(page.get('confidence', 0) for page in extracted_data) / total_pages
            
            summary = f"Document processed: {total_pages} pages, {total_text_length} characters, {avg_confidence:.2f} confidence"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Summary generation failed"
    
    def _determine_layout_type(self, rectangles: List[Dict], lines: Optional[np.ndarray]) -> str:
        """Determine document layout type"""
        try:
            if len(rectangles) > 5:
                return 'form'
            elif lines is not None and len(lines) > 10:
                return 'table'
            else:
                return 'text'
        except:
            return 'unknown'
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.layout_model:
                del self.layout_model
            if self.nlp_model:
                del self.nlp_model
            
            logger.info("Document Processor resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 