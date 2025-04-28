from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import os
import uuid

# Use headless OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # Avoids some OpenCV warnings

app = FastAPI()
model = YOLO("best.pt")  # Update path

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = f"temp_{uuid.uuid4()}{file_ext}"
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Run prediction (using OpenCV headless)
        results = model.predict(temp_path)
        output_path = f"result_{uuid.uuid4()}.jpg"
        results[0].save(output_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return FileResponse(output_path)

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "API is healthy"}