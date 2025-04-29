from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import uuid
import shutil

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
model = YOLO("best.pt")

# Create directories if they don't exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Generate unique filenames
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        temp_path = f"temp_uploads/temp_{uuid.uuid4()}{file_ext}"
        output_path = f"results/result_{uuid.uuid4()}.jpg"
        
        # Save upload
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)  # More efficient for large files
        
        # Process prediction
        results = model.predict(temp_path)
        results[0].save(output_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            headers={
                "Access-Control-Expose-Headers": "Content-Disposition",
                "Content-Disposition": f"attachment; filename=result.jpg"
            }
        )
    
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Ensure file is closed
        await file.close()

@app.get("/")
def health_check():
    return {"status": "API is healthy"}

# Cleanup function to remove old files
def cleanup_files():
    for folder in ["temp_uploads", "results"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Run cleanup on startup
cleanup_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
