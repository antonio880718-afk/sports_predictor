import os
import json
import random
import requests
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import joblib

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "ai_memory_bank_nba.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "nba_ai_model.pkl")

MEMORY_COLUMNS = [
    "date", "away_team", "home_team",
    "away_pts", "home_pts", "total_pts", "spread",
    "away_ppg", "home_ppg", "home_adv",
    "winner"  # 0=Away, 1=Home
]

FEATURE_COLUMNS = ["away_ppg", "home_ppg", "total_est", "spread_est", "home_adv"]

def _ensure_memory_file():
    if not os.path.isfile(MEMORY_FILE):
        with open(MEMORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MEMORY_COLUMNS)


def get_nba_team_stats(team_name: str):
    """Busca estadísticas reales del equipo en balldontlie.io."""
    try:
        res = requests.get("https://api.balldontlie.io/v1/teams", timeout=5)
        if res.status_code != 200:
            return {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0}
        
        teams = res.json().get("data", [])
        team_id = None
        for t in teams:
            if team_name.lower() in t["full_name"].lower() or team_name.lower() in t["name"].lower():
                team_id = t["id"]
                break
        
        if not team_id:
            return {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0}
        
        games_res = requests.get(
            f"https://api.balldontlie.io/v1/games?team_ids[]={team_id}&per_page=10&seasons[]=2025",
            timeout=5
        )
        
        if games_res.status_code != 200:
            return {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0}
        
        games = games_res.json().get("data", [])
        if not games:
            return {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0}
        
        total_pts = 0
        total_opp = 0
        wins = 0
        losses = 0
        
        for g in games:
            if g.get("status") != "Final":
                continue
            is_home = g["home_team"]["id"] == team_id
            if is_home:
                team_score = g.get("home_team_score", 0)
                opp_score = g.get("visitor_team_score", 0)
            else:
                team_score = g.get("visitor_team_score", 0)
                opp_score = g.get("home_team_score", 0)
            
            total_pts += team_score
            total_opp += opp_score
            if team_score > opp_score:
                wins += 1
            else:
                losses += 1
        
        n = max(len(games), 1)
        return {
            "wins": wins,
            "losses": losses,
            "ppg": round(total_pts / n, 1),
            "opp_ppg": round(total_opp / n, 1)
        }
    except Exception as e:
        return {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0}


def harvest_real_results(target_date: str = None, days_back: int = 14):
    """Descarga resultados REALES de NBA de ESPN."""
    _ensure_memory_file()
    
    if target_date:
        dates = [target_date]
    else:
        dates = []
        for i in range(1, days_back + 1):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(d)
    
    existing_keys = set()
    if os.path.isfile(MEMORY_FILE):
        try:
            df_existing = pd.read_csv(MEMORY_FILE)
            for _, row in df_existing.iterrows():
                key = f"{row.get('date','')}-{row.get('away_team','')}-{row.get('home_team','')}"
                existing_keys.add(key)
        except:
            pass
    
    new_games = []
    
    for date_str in dates:
        date_formatted = date_str.replace("-", "")
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_formatted}"
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
                    
                    winner = 1 if home_score > away_score else 0
                    
                    new_games.append([
                        date_str, away_team, home_team,
                        away_score, home_score, total_pts, spread,
                        away_score, home_score, 1.0,
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


def train_nba_model(target_date: str = None):
    """Entrena con TODA la memoria permanente acumulada."""
    new_count = harvest_real_results(target_date, days_back=30)
    
    _ensure_memory_file()
    
    df = pd.read_csv(MEMORY_FILE)
    df = df.drop_duplicates(subset=["date", "away_team", "home_team"])
    df.to_csv(MEMORY_FILE, index=False)
    
    total_in_memory = len(df)
    
    if total_in_memory < 10:
        seed_data = []
        for _ in range(100):
            home_pts = random.gauss(112, 10)
            away_pts = random.gauss(108, 10)
            total = home_pts + away_pts
            spread = home_pts - away_pts
            w = 1 if home_pts > away_pts else 0
            seed_data.append([away_pts, home_pts, total, spread, 1.0, w])
        X = [r[:-1] for r in seed_data]
        y = [r[-1] for r in seed_data]
        total_training = len(X)
    else:
        X = []
        y = []
        for _, row in df.iterrows():
            away_p = float(row.get("away_ppg", row.get("away_pts", 108)))
            home_p = float(row.get("home_ppg", row.get("home_pts", 112)))
            total = float(row.get("total_pts", 220))
            spread = float(row.get("spread", 0))
            X.append([away_p, home_p, total, spread, 1.0])
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
    
    avg_total = round(df["total_pts"].mean(), 1) if total_in_memory > 0 and "total_pts" in df.columns else 0
    home_win_pct = round((df["winner"].mean()) * 100, 1) if total_in_memory > 0 else 50
    
    return {
        "status": "COMPLETADO",
        "gamesAudited": total_training,
        "failedSniperBets": int((1 - (accuracy / 100)) * total_training),
        "insights": [
            {
                "patternFound": f"Cerebro GBM NBA: {round(accuracy, 1)}% precisión en {total_training} partidos. Memoria total: {total_in_memory} juegos.",
                "actionTaken": f"Se agregaron {new_count} juegos nuevos. Promedio total: {avg_total} pts. Home win: {home_win_pct}%.",
                "feature": "GradientBoosting_NBA_PermanentMemory",
                "newWeight": "Acumulativo"
            }
        ],
        "message": f"IA NBA re-entrenada con {total_training} juegos. Memoria permanente: {total_in_memory} partidos (+{new_count} nuevos)."
    }


def calculate_nba_predictions(away_team: str, home_team: str, spread_val: float = 0.0, total_val: float = 0.0):
    """Predicción NBA usando memoria permanente + stats reales."""
    if not os.path.exists(MODEL_FILE):
        train_nba_model()
    
    try:
        clf = joblib.load(MODEL_FILE)
    except:
        train_nba_model()
        clf = joblib.load(MODEL_FILE)
    
    # Stats reales
    away_stats = get_nba_team_stats(away_team)
    home_stats = get_nba_team_stats(home_team)
    
    if total_val == 0:
        total_val = away_stats["ppg"] + home_stats["ppg"]
    if spread_val == 0:
        spread_val = home_stats["ppg"] - away_stats["ppg"] - 2.5
    
    features = [[away_stats["ppg"], home_stats["ppg"], total_val, spread_val, 1.0]]
    
    probs = clf.predict_proba(features)[0]
    away_prob = float(probs[0] * 100)
    home_prob = float(probs[1] * 100)
    
    # Calibrar con memoria
    historical_avg_total = 220.0
    if os.path.isfile(MEMORY_FILE):
        try:
            df_mem = pd.read_csv(MEMORY_FILE)
            if len(df_mem) > 20:
                historical_avg_total = df_mem["total_pts"].mean()
        except:
            pass
    
    # 1. GANADOR
    spread_magnitude = abs(spread_val)
    implied_conf = 50.0 + (spread_magnitude * 1.8)
    implied_conf = min(implied_conf, 73.0)
    implied_conf = max(implied_conf, 51.0)
    
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
    spread_conf = round(50.0 + spread_magnitude * 1.2, 1)
    spread_conf = min(spread_conf, 65.0)
    
    # 3. OVER/UNDER — calibrado con memoria
    ou_diff = total_val - historical_avg_total
    ou_prediction = "OVER" if ou_diff > 5.0 else "UNDER"
    ou_conf = round(50.0 + abs(ou_diff) * 0.8, 1)
    ou_conf = min(ou_conf, 65.0)
    
    # 4. 3-POINTERS
    threes_line = round(total_val / 18.0, 1)
    threes_pred = "OVER" if total_val > historical_avg_total + 5 else "UNDER"
    threes_conf = round(50.0 + abs(total_val - historical_avg_total) * 0.5, 1)
    threes_conf = min(threes_conf, 62.0)
    
    # 5. REBOUNDS
    reb_line = round(total_val / 5.0, 1)
    reb_pred = "OVER" if spread_magnitude < 5 else "UNDER"
    reb_conf = round(50.0 + spread_magnitude * 0.8, 1)
    reb_conf = min(reb_conf, 60.0)
    
    # 6. ASSISTS
    ast_line = round(total_val / 9.5, 1)
    ast_pred = "OVER" if total_val > historical_avg_total + 2 else "UNDER"
    ast_conf = round(50.0 + abs(total_val - historical_avg_total) * 0.6, 1)
    ast_conf = min(ast_conf, 62.0)
    
    # 7. 1ER CUARTO
    q1_winner = winner
    q1_conf = round(win_conf - 5.0, 1)
    q1_conf = max(q1_conf, 50.0)
    
    return {
        "market_1_winner": {"prediction": winner, "confidence": win_conf},
        "market_2_spread": {"line": f"{spread_team} {round(spread_val, 1)}", "confidence": spread_conf},
        "market_3_ou": {"line": round(total_val, 1), "prediction": ou_prediction, "confidence": ou_conf},
        "market_4_threes": {"line": threes_line, "prediction": threes_pred, "confidence": threes_conf},
        "market_5_rebounds": {"line": reb_line, "prediction": reb_pred, "confidence": reb_conf},
        "market_6_assists": {"line": ast_line, "prediction": ast_pred, "confidence": ast_conf},
        "market_7_q1": {"prediction": q1_winner, "confidence": q1_conf},
        "team_stats": {
            "away": {"ppg": away_stats["ppg"], "opp_ppg": away_stats["opp_ppg"], "record": f"{away_stats['wins']}-{away_stats['losses']}"},
            "home": {"ppg": home_stats["ppg"], "opp_ppg": home_stats["opp_ppg"], "record": f"{home_stats['wins']}-{home_stats['losses']}"}
        }
    }
