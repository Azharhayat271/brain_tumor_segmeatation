from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import os
import uuid

app = FastAPI()

# Load model - ensure 'best.pt' is in your repository
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Generate unique filenames
        temp_path = f"temp_{uuid.uuid4()}.jpg"
        output_path = f"result_{uuid.uuid4()}.jpg"
        
        # Save upload
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process prediction
        results = model.predict(temp_path)
        results[0].save(output_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return FileResponse(output_path)
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "API is healthy"}

# Add this for Render compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
