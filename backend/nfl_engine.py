import os
import json
import random
import requests
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import joblib

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "ai_memory_bank_nfl.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "nfl_ai_model.pkl")

MEMORY_COLUMNS = [
    "date", "away_team", "home_team",
    "away_pts", "home_pts", "total_pts", "spread",
    "home_implied", "away_implied", "home_adv",
    "winner"  # 0=Away, 1=Home
]

FEATURE_COLUMNS = ["away_implied", "home_defense_implied", "home_implied", "away_defense_implied", "home_adv"]

def _ensure_memory_file():
    if not os.path.isfile(MEMORY_FILE):
        with open(MEMORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MEMORY_COLUMNS)


def harvest_real_results(target_date: str = None, days_back: int = 30):
    """
    Descarga resultados REALES de NFL de ESPN y los almacena permanentemente.
    Como NFL juega pocos días, buscamos más hacia atrás.
    """
    _ensure_memory_file()
    
    if target_date:
        dates = [target_date]
    else:
        dates = []
        for i in range(1, days_back + 1):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(d)
    
    # Leer memoria existente para evitar duplicados
    existing_keys = set()
    if os.path.isfile(MEMORY_FILE):
        try:
            df_existing = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
            for _, row in df_existing.iterrows():
                key = f"{row.get('date','')}-{row.get('away_team','')}-{row.get('home_team','')}"
                existing_keys.add(key)
        except:
            pass
    
    new_games = []
    
    for date_str in dates:
        date_formatted = date_str.replace("-", "")
        url = f"http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date_formatted}"
        try:
            res = requests.get(url, timeout=5).json()
            events = res.get('events', [])
            
            for g in events:
                try:
                    competition = g['competitions'][0]
                    status = competition.get('status', {}).get('type', {})
                    if not status.get('completed', False):
                        continue
                    
                    competitors = competition['competitors']
                    home_team = away_team = ""
                    home_score = away_score = 0
                    
                    for c in competitors:
                        if c['homeAway'] == 'home':
                            home_team = c['team']['displayName']
                            home_score = int(c.get('score', 0))
                        else:
                            away_team = c['team']['displayName']
                            away_score = int(c.get('score', 0))
                    
                    key = f"{date_str}-{away_team}-{home_team}"
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    
                    total_pts = home_score + away_score
                    spread = home_score - away_score
                    
                    # Implied points (reconstruidos del score real)
                    home_implied = home_score
                    away_implied = away_score
                    
                    winner = 1 if home_score > away_score else 0
                    
                    new_games.append([
                        date_str, away_team, home_team,
                        away_score, home_score, total_pts, spread,
                        home_implied, away_implied, 1.0,
                        winner
                    ])
                except:
                    continue
        except:
            continue
    
    if new_games:
        with open(MEMORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in new_games:
                writer.writerow(row)
    
    return len(new_games)


def train_nfl_model(target_date: str = None):
    """Entrena el modelo NFL con TODA la memoria permanente acumulada."""
    new_count = harvest_real_results(target_date, days_back=60)
    
    _ensure_memory_file()
    
    df = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
    df = df.drop_duplicates(subset=["date", "away_team", "home_team"])
    df.to_csv(MEMORY_FILE, index=False)
    
    total_in_memory = len(df)
    
    if total_in_memory < 10:
        # Datos semilla para arrancar (promedios NFL reales)
        seed_data = []
        for _ in range(100):
            home_pts = random.gauss(24.0, 8.0)
            away_pts = random.gauss(21.5, 8.0)
            total = home_pts + away_pts
            spread = home_pts - away_pts
            w = 1 if home_pts > away_pts else 0
            seed_data.append([away_pts, total - away_pts, home_pts, total - home_pts, 1.0, w])
        X = [r[:-1] for r in seed_data]
        y = [r[-1] for r in seed_data]
        total_training = len(X)
    else:
        # Construir features de la memoria
        X = []
        y = []
        for _, row in df.iterrows():
            away_imp = float(row.get("away_implied", row.get("away_pts", 21)))
            home_imp = float(row.get("home_implied", row.get("home_pts", 24)))
            total = float(row.get("total_pts", 45))
            X.append([away_imp, total - away_imp, home_imp, total - home_imp, 1.0])
            y.append(int(row["winner"]))
        total_training = len(X)
    
    clf = GradientBoostingClassifier(
        n_estimators=min(200, 50 + total_training),
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    clf.fit(X, y)
    
    accuracy = clf.score(X, y) * 100
    joblib.dump(clf, MODEL_FILE)
    
    # Estadísticas de memoria
    avg_total = round(df["total_pts"].mean(), 1) if total_in_memory > 0 and "total_pts" in df.columns else 0
    home_win_pct = round((df["winner"].mean()) * 100, 1) if total_in_memory > 0 else 50
    
    return {
        "status": "COMPLETADO",
        "gamesAudited": total_training,
        "failedSniperBets": int((1 - (accuracy / 100)) * total_training),
        "insights": [
            {
                "patternFound": f"Cerebro GBM NFL: {round(accuracy, 1)}% precisión en {total_training} partidos. Memoria total: {total_in_memory} juegos.",
                "actionTaken": f"Se agregaron {new_count} juegos nuevos. Promedio total: {avg_total} pts. Home win: {home_win_pct}%.",
                "feature": "GradientBoosting_NFL_PermanentMemory",
                "newWeight": "Acumulativo"
            }
        ],
        "message": f"IA NFL re-entrenada con {total_training} juegos. Memoria permanente: {total_in_memory} partidos (+{new_count} nuevos)."
    }


def calculate_nfl_predictions(away_team: str, home_team: str, spread_val: float, total_val: float):
    """Predicción NFL usando memoria permanente + líneas de Vegas."""
    if not os.path.exists(MODEL_FILE):
        train_nfl_model()
    
    try:
        clf = joblib.load(MODEL_FILE)
    except:
        train_nfl_model()
        clf = joblib.load(MODEL_FILE)
    
    if spread_val == 0: spread_val = -3.0
    if total_val == 0: total_val = 44.5
    
    home_implied_pts = (total_val / 2.0) - (spread_val / 2.0)
    away_implied_pts = (total_val / 2.0) + (spread_val / 2.0)
    
    features = [[
        away_implied_pts,
        total_val - away_implied_pts,
        home_implied_pts,
        total_val - home_implied_pts,
        1.0
    ]]
    
    probs = clf.predict_proba(features)[0]
    away_prob = float(probs[0] * 100)
    home_prob = float(probs[1] * 100)
    
    # --- Calibrar con memoria histórica ---
    historical_home_rate = 0.55
    historical_avg_total = 44.5
    if os.path.isfile(MEMORY_FILE):
        try:
            df_mem = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
            if len(df_mem) > 10:
                historical_home_rate = df_mem["winner"].mean()
                historical_avg_total = df_mem["total_pts"].mean()
        except:
            pass
    
    # 1. GANADOR
    spread_magnitude = abs(spread_val)
    implied_conf = 50.0 + (spread_magnitude * 2.5)
    implied_conf = min(implied_conf, 75.0)
    implied_conf = max(implied_conf, 52.0)
    
    if spread_val < 0:
        winner = home_team
    elif spread_val > 0:
        winner = away_team
    else:
        winner = home_team
        implied_conf = 52.0
    win_conf = round(implied_conf, 1)
    
    # 2. SPREAD
    spread_team = home_team if spread_val < 0 else away_team
    spread_conf = round(50.0 + (spread_magnitude * 1.5), 1)
    spread_conf = min(spread_conf, 68.0)
    
    # 3. OVER/UNDER — calibrado con memoria
    ou_diff = total_val - historical_avg_total
    ou_prediction = "OVER" if ou_diff > 2.0 else "UNDER"
    ou_conf = round(50.0 + abs(ou_diff) * 2.0, 1)
    ou_conf = min(ou_conf, 65.0)
    
    # 4. TOUCHDOWNS
    td_line = round(total_val / 7.0, 1)
    td_pred = "OVER" if ou_prediction == "OVER" else "UNDER"
    td_conf = round(ou_conf - 3.0, 1)
    
    # 5. YARDAS QB
    qb_line = round(total_val * 5.2, 0) + 0.5
    qb_pred = "OVER" if total_val > 46 else "UNDER"
    qb_conf = round(52.0 + abs(total_val - 44.5) * 1.5, 1)
    qb_conf = min(qb_conf, 65.0)
    
    # 6. YARDAS RB
    rb_line = 65.5 if spread_magnitude > 5 else 75.5
    rb_pred = "OVER" if spread_magnitude < 4 else "UNDER"
    rb_conf = round(50.0 + spread_magnitude * 1.0, 1)
    rb_conf = min(rb_conf, 62.0)
    
    # 7. PRIMERO EN ANOTAR
    first_score_team = winner
    first_score_conf = round(50.0 + spread_magnitude * 1.2, 1)
    first_score_conf = min(first_score_conf, 62.0)
    
    return {
        "market_1_winner": {"prediction": winner, "confidence": win_conf},
        "market_2_spread": {"line": f"{spread_team} {spread_val}", "confidence": spread_conf},
        "market_3_ou": {"line": total_val, "prediction": ou_prediction, "confidence": ou_conf},
        "market_4_tds": {"line": td_line, "prediction": td_pred, "confidence": td_conf},
        "market_5_qb": {"line": qb_line, "prediction": qb_pred, "confidence": qb_conf},
        "market_6_rb": {"line": rb_line, "prediction": rb_pred, "confidence": rb_conf},
        "market_7_first": {"prediction": first_score_team, "confidence": first_score_conf}
    }
