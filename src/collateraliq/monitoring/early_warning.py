"""
Early Warning System for clients.
Uses Logistic Regression to predict margin breach probability.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class EarlyWarningModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced")
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """Train on historical margin limit breaches."""
        features = df[["coverage_ratio", "concentration_score", "volatility_index"]]
        labels = df["breached"]
        self.model.fit(features, labels)
        self.is_trained = True
        
    def predict(self, current_data: pd.DataFrame) -> pd.Series:
        """Predict probability of breach in next 5 days."""
        if not self.is_trained:
            # Fallback heuristic
            return current_data["coverage_ratio"].apply(lambda x: 0.9 if x < 1.1 else 0.1)
            
        features = current_data[["coverage_ratio", "concentration_score", "volatility_index"]]
        probs = self.model.predict_proba(features)[:, 1]
        return pd.Series(probs, index=current_data.index)

def score_clients(db_path: str) -> pd.DataFrame:
    """Generate explainable early warning scores based on DB metrics."""
    from collateraliq.data.db import get_engine
    engine = get_engine(db_path)
    
    with engine.begin() as conn:
        clients = pd.read_sql("SELECT client_id, archetype FROM clients", conn)
        # Fetch latest exposures and IM
        exp = pd.read_sql("SELECT client_id, exposure_inr, limit_util_pct FROM exposures WHERE date = (SELECT MAX(date) FROM exposures)", conn)
        im = pd.read_sql("SELECT client_id, im_total_inr, addon_concentration_inr FROM im_results WHERE product_type = 'Portfolio' AND date = (SELECT MAX(date) FROM im_results)", conn)
        vm = pd.read_sql("SELECT client_id, vm_held_inr FROM vm_ledger WHERE date = (SELECT MAX(date) FROM vm_ledger)", conn)
        
    df = clients.merge(exp, on="client_id", how="left").merge(im, on="client_id", how="left").merge(vm, on="client_id", how="left").fillna(0)
    
    scores = []
    for _, row in df.iterrows():
        score = 0.0
        reasons = []
        
        # 1. Coverage Ratio
        total_collateral = row["im_total_inr"] + row["vm_held_inr"]
        exposure = max(row["exposure_inr"], 1.0)
        coverage = total_collateral / exposure
        if exposure > 1000:
            if coverage < 1.0:
                score += 0.4
                reasons.append("Under-collateralised")
            elif coverage < 1.1:
                score += 0.15
                reasons.append("Low coverage buffer")
                
        # 2. Concentration
        concentration = row["addon_concentration_inr"]
        if concentration > 1000000:
            score += 0.3
            reasons.append("High position concentration")
            
        # 3. Limit Utilisation (proxy for volatility/stress)
        util = row["limit_util_pct"]
        if util > 80:
            score += 0.25
            reasons.append("Credit limit >80%")
            
        prob = min(score + 0.05, 0.95) # Base 5% noise
        cat = "High" if prob > 0.4 else ("Medium" if prob > 0.15 else "Low")
        scores.append({"client_id": row["client_id"], "breach_probability": prob, "risk_category": cat, "key_drivers": ", ".join(reasons) if reasons else "Normal"})
        
    return pd.DataFrame(scores)
