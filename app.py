from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import uuid
import shutil
import cv2

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
model = YOLO("best.pt")  # Replace with your custom-trained model path

# Create required directories
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tumor_type: str = Form(...) 
):
    try:
        # Save file
        file_ext = os.path.splitext(file.filename)[1] or ".jpg"
        temp_path = f"temp_uploads/temp_{uuid.uuid4()}{file_ext}"
        output_path = f"results/result_{uuid.uuid4()}.jpg"

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run YOLO prediction
        results = model.predict(temp_path)
        result = results[0]

        # Get image with mask (no labels)
        img = result.plot(labels=False)

        # Add custom tumor label manually
        if result.masks is not None:
            # Place text near the first mask bounding box
            first_box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = first_box[0], first_box[1]

            cv2.putText(
                img,
                f"{tumor_type}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Save the final image
        cv2.imwrite(output_path, img)
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
        await file.close()

@app.get("/")
def health_check():
    return {"status": "API is healthy"}

def cleanup_files():
    for folder in ["temp_uploads", "results"]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

cleanup_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
