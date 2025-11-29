"""
Run the FastAPI application from the src directory.
This allows the API to import modules from both api/ and root src/ level (e.g., quizlet_rag.py).

Usage:
    cd src
    python run_api.py
    
Or with uvicorn directly:
    cd src
    uvicorn api.main:app --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
