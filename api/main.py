import uvicorn
from transformers import pipeline
from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI(
    title="Serving Meta's OPT",
    description="API to serve dynamically OPT model",
    version="1.0",
    openapi_url="/api/v1/openapi.json",
)

default_size = "350m"

pipe = pipeline("text-generation", f"AlekseyKorshuk/opt-{default_size}")


@app.get("/", include_in_schema=False)
async def home():
    """
    Home endpoint to redirect to docs.
    """
    return RedirectResponse("/docs")


@app.put("/predict", include_in_schema=True, tags=["serving"])
async def predict(input_text: str):
    """
    Predict endpoint.
    Parameters
    ----------
    input_text: str
        Input string to generate.
    -------
    """
    return pipe(input_text)


@app.put("/select_model", include_in_schema=True, tags=["serving"])
async def select_model(model_size: str):
    """
    Update model endpoint.
    """
    global pipe
    try:
        pipe = pipeline("text-generation", f"AlekseyKorshuk/opt-{model_size}")
    except Exception as ex:
        pipe = pipeline("text-generation", f"AlekseyKorshuk/opt-{default_size}")
        return {"error": str(ex)}

    return {"message": "Model has been updated."}


@app.get("/healthz", status_code=200, include_in_schema=True, tags=["monitoring"])
async def healthz():
    """
    Healthz endpoint.
    """
    return {"status": "ok"}


@app.get("/readyz", status_code=200, include_in_schema=True, tags=["monitoring"])
async def readyz():
    """
    Readyz endpoint.
    """
    return {"status": "ready"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", reload=True)
