from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extractor import FeatureExtractor
from models.detector import SybilDetector


app = FastAPI(
    title="Sybil Wallet Detection API",
    description="Production API for detecting Sybil wallets on Ethereum",
    version="1.0.0"
)


class WalletData(BaseModel):
    address: str
    transaction_count: int = Field(ge=0)
    unique_interactions: int = Field(ge=0)
    avg_transaction_value: float = Field(ge=0)
    transaction_value_std: float = Field(ge=0)
    current_balance: float = Field(ge=0)
    max_balance: float = Field(ge=0)
    gas_price_std: float = Field(ge=0)
    avg_time_between_tx: float = Field(ge=0)
    contract_interaction_count: int = Field(ge=0)
    nft_transfer_count: int = Field(ge=0)
    erc20_token_count: int = Field(ge=0)
    incoming_tx_count: int = Field(ge=0)
    outgoing_tx_count: int = Field(ge=0)
    creation_timestamp: float


class BatchWalletRequest(BaseModel):
    wallets: List[WalletData]
    graph_edges: Optional[List[Dict[str, str]]] = None


class PredictionResponse(BaseModel):
    address: str
    is_sybil: bool
    confidence: float
    risk_score: float
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    sybil_detected: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    model_type: str
    version: str
    last_trained: str
    feature_count: int
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


detector = None
feature_extractor = None
start_time = datetime.now()


@app.on_event("startup")
async def load_models():
    global detector, feature_extractor

    try:
        detector = SybilDetector.load('models/sybil_detector.pkl')
        feature_extractor = joblib.load('models/feature_extractor.pkl')
    except FileNotFoundError:
        detector = None
        feature_extractor = None


@app.get("/", response_model=Dict)
async def root():
    return {
        "service": "Sybil Wallet Detection API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = (datetime.now() - start_time).total_seconds()

    return HealthResponse(
        status="healthy" if detector is not None else "degraded",
        model_loaded=detector is not None,
        uptime_seconds=uptime
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type="Ensemble (XGBoost + LightGBM + RandomForest)",
        version="1.0.0",
        last_trained=datetime.now().isoformat(),
        feature_count=len(feature_extractor.feature_names) if feature_extractor else 0,
        threshold=0.75
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_wallet(wallet: WalletData):
    if detector is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = datetime.now()

    try:
        df = pd.DataFrame([wallet.dict()])

        G = nx.DiGraph()
        G.add_node(wallet.address)

        X = feature_extractor.transform(df, G)

        y_proba = detector.predict_proba(X)[0]
        confidence = float(y_proba[1])
        is_sybil = confidence > 0.75

        risk_score = min(confidence * 100, 100.0)

        processing_time = (datetime.now() - start).total_seconds() * 1000

        return PredictionResponse(
            address=wallet.address,
            is_sybil=is_sybil,
            confidence=confidence,
            risk_score=risk_score,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_wallets(request: BatchWalletRequest):
    if detector is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = datetime.now()

    try:
        df = pd.DataFrame([w.dict() for w in request.wallets])

        G = nx.DiGraph()
        for wallet in request.wallets:
            G.add_node(wallet.address)

        if request.graph_edges:
            for edge in request.graph_edges:
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))

        X = feature_extractor.transform(df, G)

        y_proba = detector.predict_proba(X)

        predictions = []
        sybil_count = 0

        for idx, wallet in enumerate(request.wallets):
            confidence = float(y_proba[idx][1])
            is_sybil = confidence > 0.75

            if is_sybil:
                sybil_count += 1

            predictions.append(PredictionResponse(
                address=wallet.address,
                is_sybil=is_sybil,
                confidence=confidence,
                risk_score=min(confidence * 100, 100.0),
                processing_time_ms=0
            ))

        processing_time = (datetime.now() - start).total_seconds() * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            sybil_detected=sybil_count,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats", response_model=Dict)
async def get_statistics():
    return {
        "total_predictions": 0,
        "sybil_detected": 0,
        "average_confidence": 0.0,
        "uptime_hours": (datetime.now() - start_time).total_seconds() / 3600
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
