"""
NVIDIA HPC Manager for FinDoc AI Platform
Handles GPU initialization, monitoring, and optimization
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import psutil

try:
    import pynvml
    import torch
    try:
        import cupy as cp  # type: ignore
        CUPY_AVAILABLE = True
    except ImportError:
        CUPY_AVAILABLE = False
        logging.warning("CuPy not available. GPU memory management will be limited.")
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    CUPY_AVAILABLE = False
    logging.warning("NVIDIA libraries not available. Running in CPU-only mode.")

from backend.core.config import settings

logger = logging.getLogger(__name__)


class NVIDIAHPCManager:
    """Manages NVIDIA HPC resources and GPU operations"""
    
    def __init__(self):
        self.initialized = False
        self.gpu_count = 0
        self.gpu_info = {}
        self.memory_pools = {}
        self.streams = {}
        
    async def initialize(self) -> None:
        """Initialize NVIDIA HPC manager"""
        try:
            if not NVIDIA_AVAILABLE:
                logger.warning("NVIDIA libraries not available. Initializing CPU-only mode.")
                self.initialized = True
                return
                
            # Initialize NVML
            try:
                pynvml.nvmlInit()
            except Exception as e:
                logger.warning(f"NVIDIA NVML initialization failed: {e}. Running in CPU-only mode.")
                self.initialized = True
                return
            
            # Get GPU count and info
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"Found {self.gpu_count} NVIDIA GPU(s)")
            except Exception as e:
                logger.warning(f"Could not get GPU count: {e}. Running in CPU-only mode.")
                self.gpu_count = 0
                self.initialized = True
                return
            
            # Initialize GPU information
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    self.gpu_info[i] = {
                        'name': name,
                        'total_memory': int(memory_info.total),
                        'free_memory': int(memory_info.free),
                        'used_memory': int(memory_info.used),
                        'handle': handle
                    }
                    
                    logger.info(f"GPU {i}: {name} - {int(memory_info.total) // 1024**3}GB total")
                except Exception as e:
                    logger.warning(f"Could not initialize GPU {i}: {e}")
                    continue
            
            # Set PyTorch device
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # Use first GPU
                torch.backends.cudnn.benchmark = True
                if settings.nvidia_mixed_precision:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                logger.info("PyTorch CUDA initialized successfully")
            else:
                logger.info("CUDA not available. Using CPU for PyTorch.")
            
            # Initialize CuPy memory pools
            if CUPY_AVAILABLE and cp.cuda.is_available():
                for i in range(self.gpu_count):
                    with cp.cuda.Device(i):
                        self.memory_pools[i] = cp.get_default_memory_pool()
                        self.streams[i] = cp.cuda.Stream()
                
                logger.info("CuPy memory pools initialized")
            elif not CUPY_AVAILABLE:
                logger.info("CuPy not available - GPU memory management disabled")
            
            self.initialized = True
            logger.info("✅ NVIDIA HPC Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA HPC Manager: {e}")
            # Don't raise the exception, just log it and continue in CPU-only mode
            logger.info("Continuing in CPU-only mode...")
            self.initialized = True
    
    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU information"""
        if not self.initialized:
            return {"error": "NVIDIA HPC Manager not initialized"}
        
        if not NVIDIA_AVAILABLE:
            return {"mode": "cpu_only", "message": "Running in CPU-only mode"}
        
        try:
            gpu_status = {}
            
            for i in range(self.gpu_count):
                handle = self.gpu_info[i]['handle']
                
                # Get current memory usage
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                except:
                    power = None
                
                gpu_status[i] = {
                    'name': self.gpu_info[i]['name'],
                    'memory': {
                        'total': int(memory_info.total),
                        'used': int(memory_info.used),
                        'free': int(memory_info.free),
                        'utilization': (int(memory_info.used) / int(memory_info.total)) * 100
                    },
                    'utilization': utilization.gpu,
                    'temperature': temperature,
                    'power': power,
                    'timestamp': time.time()
                }
            
            return {
                'gpu_count': self.gpu_count,
                'gpus': gpu_status,
                'system_memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used,
                    'percent': psutil.virtual_memory().percent
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return {"error": str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        gpu_info = await self.get_gpu_info()
        
        return {
            'initialized': self.initialized,
            'nvidia_available': NVIDIA_AVAILABLE,
            'gpu_info': gpu_info,
            'settings': {
                'gpu_enabled': settings.nvidia_gpu_enabled,
                'memory_fraction': settings.nvidia_gpu_memory_fraction,
                'mixed_precision': settings.nvidia_mixed_precision
            }
        }
    
    async def allocate_memory(self, size: int, gpu_id: int = 0) -> Optional[Any]:
        """Allocate GPU memory"""
        if not self.initialized or not NVIDIA_AVAILABLE or not CUPY_AVAILABLE:
            return None
        
        try:
            if gpu_id in self.memory_pools:
                with cp.cuda.Device(gpu_id):
                    memory = self.memory_pools[gpu_id].malloc(size)
                    return memory
        except Exception as e:
            logger.error(f"Error allocating GPU memory: {e}")
            return None
    
    async def free_memory(self, memory: Any, gpu_id: int = 0) -> bool:
        """Free GPU memory"""
        if not self.initialized or not NVIDIA_AVAILABLE or not CUPY_AVAILABLE:
            return False
        
        try:
            if gpu_id in self.memory_pools:
                with cp.cuda.Device(gpu_id):
                    self.memory_pools[gpu_id].free(memory)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error freeing GPU memory: {e}")
            return False
    
    async def get_memory_usage(self, gpu_id: int = 0) -> Dict[str, Any]:
        """Get memory usage for specific GPU"""
        if not self.initialized or not NVIDIA_AVAILABLE:
            return {"error": "NVIDIA not available"}
        
        try:
            if gpu_id >= self.gpu_count:
                return {"error": f"GPU {gpu_id} not found"}
            
            handle = self.gpu_info[gpu_id]['handle']
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'total': int(memory_info.total),
                'used': int(memory_info.used),
                'free': int(memory_info.free),
                'utilization': (int(memory_info.used) / int(memory_info.total)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"error": str(e)}
    
    async def optimize_for_processing(self) -> Dict[str, Any]:
        """Optimize GPU settings for document processing"""
        if not self.initialized or not NVIDIA_AVAILABLE:
            return {"message": "Running in CPU-only mode"}
        
        try:
            optimizations = {}
            
            # Set memory fraction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(settings.nvidia_gpu_memory_fraction)
                optimizations['memory_fraction'] = settings.nvidia_gpu_memory_fraction
            
            # Enable mixed precision
            if settings.nvidia_mixed_precision:
                optimizations['mixed_precision'] = True
            
            # Set CuPy memory pool limits
            for i in range(self.gpu_count):
                if i in self.memory_pools:
                    pool = self.memory_pools[i]
                    pool.set_limit(size=2**30)  # 1GB limit
                    optimizations[f'gpu_{i}_memory_limit'] = "1GB"
            
            logger.info("GPU optimizations applied for document processing")
            return optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing GPU: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup NVIDIA HPC resources"""
        try:
            if NVIDIA_AVAILABLE and self.initialized:
                # Clear PyTorch cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear CuPy memory pools
                for pool in self.memory_pools.values():
                    pool.free_all_blocks()
                
                # Shutdown NVML
                pynvml.nvmlShutdown()
                
                logger.info("NVIDIA HPC resources cleaned up")
            
            self.initialized = False
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if self.initialized:
            asyncio.create_task(self.cleanup()) 