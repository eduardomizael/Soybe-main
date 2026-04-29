from backend.routes.inference_routes import classify_image

# Converter bytes -> array NumPy -> Imagem OpenCV
def run_inference(model_name: str, imagem_bytes: bytes, weight_filename: str = None) -> dict:

    # Enviando bytes para a rota/modelo (inference_routes.py)
    resultado = classify_image(imagem_bytes, model_name, weight_filename)
    
    resultado = {
        "model_name": model_name,
        "label": resultado["class_name"],
        "confidence": resultado["confidence"]
    }

    return resultado
