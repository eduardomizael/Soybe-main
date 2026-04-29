import os
import re
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from backend.schemas import InferenceResponse
from backend.services.inference_service import run_inference
from backend.routes.training_routes import router as training_router
from starlette.requests import Request

app = FastAPI()
app.include_router(training_router, prefix="/training", tags=["training"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # True: permite cookies, False: não permite cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/inferencia")
async def options_inferencia():
    return Response(status_code=200)

# Endpoint de inferência (POST)
@app.post("/inferencia", response_model=List[InferenceResponse])
async def inferencia(
    request: Request,
    model_name: str = Form(...),
    weight_filename: str = Form(None),
    files: List[UploadFile] = File(...)
):
    results = []

    for index, f in enumerate(files):
        if await request.is_disconnected():
            print("Requisição abortada pelo cliente.")
            break

        contents = await f.read()
        try:
            prediction = run_inference(model_name, contents, weight_filename)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Lista de resultados para cada imagem enviada
        results.append({
            "index": index,
            "filename": f.filename,
            "confidence": prediction["confidence"],
            "classification": prediction["label"],
            "model_used": prediction["model_name"],
        })

        print(f"[{index + 1}] { f.filename} → {prediction['label']} ({prediction['confidence']:.4f})")

        await f.close()
 
    # Ordena por número no nome do arquivo (ex: "Broken-32/997.jpg" → 997)
    def extract_number(filename: str) -> int:
        base = os.path.basename(filename)
        match = re.search(r"\d+", base)
        return int(match.group()) if match else -1

    results.sort(key=lambda r: extract_number(r["filename"]))

    return results

@app.get("/home")
async def root():
    return {"message": "API de Inferência de Imagens está rodando!"}
