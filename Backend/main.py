# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from inference_model import predict, validate_image
from PIL import Image
import io
import logging
from typing import Dict, Any
from pydantic import BaseModel
import time
import asyncio
from starlette.requests import Request
from starlette.responses import Response

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Response models
class PredictionResponse(BaseModel):
    class_name: str
    class_description: str
    probabilities: Dict[str, float]
    processing_time: float
    confidence_score: float

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    timestamp: float

def format_class_name(class_name: str) -> str:
    """Format class name by replacing underscores with spaces and capitalizing words."""
    return ' '.join(word.capitalize() for word in class_name.split('_'))

app = FastAPI(
    title="Brain Tumor Classification API",
    description="A quantum-enhanced deep learning API for brain tumor classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", tags=["Health"])
async def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Brain Tumor Classification API",
        "version": "1.0.0"
    }

@app.post(
    "/predict/",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Prediction"]
)
async def predict_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Predict brain tumor type from an uploaded brain scan image.
    
    Args:
        file: The brain scan image file (JPG, PNG, or JPEG)
        
    Returns:
        PredictionResponse: The prediction results including class name, description,
                          probabilities, processing time, and confidence score
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG, PNG, or JPEG)"
            )
        
        logger.info(f"Received image: {file.filename}, content_type: {file.content_type}")
        
        # Read and validate image
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            validate_image(image)
            logger.info("Successfully opened and validated image")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Perform prediction
        try:
            result = predict(image)
            logger.info(f"Prediction completed successfully")
            
            # Format class names in probabilities
            result["probabilities"] = {
                format_class_name(k): v for k, v in result["probabilities"].items()
            }
            
            # Add cleanup task
            if background_tasks:
                background_tasks.add_task(lambda: image.close())
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with custom response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"ERR_{exc.status_code}",
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with custom response format."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "ERR_500",
            "timestamp": time.time()
        }
    )
