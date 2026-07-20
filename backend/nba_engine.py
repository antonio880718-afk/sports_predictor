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

_STANDINGS_CACHE = None

def _fetch_standings():
    global _STANDINGS_CACHE
    if _STANDINGS_CACHE is not None:
        return _STANDINGS_CACHE
    
    _STANDINGS_CACHE = {}
    try:
        url = "http://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return _STANDINGS_CACHE
            
        data = res.json()
        
        def _find_entries(node):
            if isinstance(node, dict):
                if 'entries' in node and isinstance(node['entries'], list):
                    yield node['entries']
                for key, value in node.items():
                    yield from _find_entries(value)
            elif isinstance(node, list):
                for item in node:
                    yield from _find_entries(item)
                    
        for entries_list in _find_entries(data):
            for entry in entries_list:
                team_name = entry.get('team', {}).get('displayName', '')
                stats = entry.get('stats', [])
                
                team_data = {
                    "wins": 0,
                    "losses": 0,
                    "winPercent": 0.5,
                    "avgPointsFor": 110.0,
                    "avgPointsAgainst": 110.0,
                    "differential": 0.0,
                    "streak": 0.0
                }
                
                for stat in stats:
                    name = stat.get("name")
                    val = stat.get("value", 0)
                    if val is None:
                        continue
                    if name in team_data:
                        team_data[name] = float(val)
                        
                if team_name:
                    _STANDINGS_CACHE[team_name.lower()] = team_data

    except Exception as e:
        pass
    
    return _STANDINGS_CACHE


def _ensure_memory_file():
    if not os.path.isfile(MEMORY_FILE):
        with open(MEMORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MEMORY_COLUMNS)

def get_nba_team_stats(team_name: str):
    """Busca estadísticas reales del equipo en ESPN Standings API."""
    cache = _fetch_standings()
    
    stats = {"wins": 20, "losses": 20, "ppg": 110.0, "opp_ppg": 110.0, "winPercent": 0.5, "differential": 0.0}
    
    best_match = None
    if cache:
        for name in cache.keys():
            if team_name.lower() in name or name in team_name.lower():
                best_match = name
                break
                
    if best_match:
        data = cache.get(best_match, {})
        stats["wins"] = int(data.get("wins", 0))
        stats["losses"] = int(data.get("losses", 0))
        stats["ppg"] = round(data.get("avgPointsFor", 110.0), 1)
        stats["opp_ppg"] = round(data.get("avgPointsAgainst", 110.0), 1)
        stats["winPercent"] = data.get("winPercent", 0.5)
        stats["differential"] = data.get("differential", 0.0)
        
    return stats


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
            df_existing = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
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
                        if c.get('homeAway') == 'home':
                            home_team = c.get('team', {}).get('displayName', '')
                            home_score = int(c.get('score', 0))
                        else:
                            away_team = c.get('team', {}).get('displayName', '')
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
    
    df = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
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
    """SIGNAL-BASED COMPOSITE prediction system using ESPN API."""
    if not os.path.exists(MODEL_FILE):
        train_nba_model()
    
    # Stats reales
    away_stats = get_nba_team_stats(away_team)
    home_stats = get_nba_team_stats(home_team)
    
    # Calculate Signals
    signals = []
    
    # Signal 1: Win Percentage (Weight: 0.30)
    h_wins = home_stats.get('wins', 0)
    h_losses = home_stats.get('losses', 0)
    a_wins = away_stats.get('wins', 0)
    a_losses = away_stats.get('losses', 0)
    
    h_winpct = h_wins / (h_wins + h_losses) if (h_wins + h_losses) > 0 else 0.5
    a_winpct = a_wins / (a_wins + a_losses) if (a_wins + a_losses) > 0 else 0.5
    s1_score = max(-1.0, min(1.0, (h_winpct - a_winpct) * 2.0))
    signals.append({"name": "Win Percentage", "score": round(s1_score, 3), "weight": 0.30})
    
    # Signal 2: Point Differential per Game (Weight: 0.25)
    h_diff = home_stats.get('ppg', 0) - home_stats.get('opp_ppg', 0)
    a_diff = away_stats.get('ppg', 0) - away_stats.get('opp_ppg', 0)
    s2_score = max(-1.0, min(1.0, (h_diff - a_diff) / 10.0))
    signals.append({"name": "Point Differential", "score": round(s2_score, 3), "weight": 0.25})
    
    # Signal 3: Home Advantage (Weight: 0.15)
    # Fixed: +0.14 composite contribution
    s3_score = 0.9333  # 0.9333 * 0.15 = ~0.14
    signals.append({"name": "Home Advantage", "score": round(s3_score, 3), "weight": 0.15})
    
    # Signal 4: Scoring Power (Weight: 0.15)
    h_ppg = home_stats.get('ppg', 0)
    a_ppg = away_stats.get('ppg', 0)
    s4_score = max(-1.0, min(1.0, (h_ppg - a_ppg) / 10.0))
    signals.append({"name": "Scoring Power", "score": round(s4_score, 3), "weight": 0.15})
    
    # Signal 5: Defensive Rating (Weight: 0.15)
    h_opp_ppg = home_stats.get('opp_ppg', 0)
    a_opp_ppg = away_stats.get('opp_ppg', 0)
    s5_score = max(-1.0, min(1.0, (a_opp_ppg - h_opp_ppg) / 8.0))
    signals.append({"name": "Defensive Rating", "score": round(s5_score, 3), "weight": 0.15})
    
    # Winner
    composite = sum(sig["score"] * sig["weight"] for sig in signals)
    
    if composite > 0:
        winner = home_team
    else:
        winner = away_team
        
    prob = 50 + abs(composite) * 22
    win_conf = round(max(50.0, min(75.0, prob)), 1)
    
    # Over/Under
    projected_total = h_ppg + a_ppg
    ou_line = round(projected_total * 2) / 2
    
    if projected_total > ou_line + 2:
        ou_prediction = "OVER"
        ou_conf = 60.0
    elif projected_total < ou_line - 2:
        ou_prediction = "UNDER"
        ou_conf = 60.0
    else:
        ou_prediction = "OVER" if projected_total >= ou_line else "UNDER"
        ou_conf = 52.0
        
    # Spread
    projected_spread_val = (h_ppg - h_opp_ppg) - (a_ppg - a_opp_ppg) + 2.5
    spread_val = -projected_spread_val
    spread_line = round(spread_val * 2) / 2
    spread_team = home_team if spread_line < 0 else away_team
    spread_conf = round(50.0 + abs(projected_spread_val) * 1.2, 1)
    spread_conf = min(spread_conf, 65.0)
    
    # 4. 3-POINTERS
    threes_line = round(projected_total / 18.0, 1)
    threes_pred = "OVER" if h_ppg + a_ppg > 220 else "UNDER"
    threes_conf = 55.0
    
    # 5. REBOUNDS
    reb_line = round(projected_total / 5.0, 1)
    reb_pred = "OVER" if abs(projected_spread_val) < 5 else "UNDER"
    reb_conf = 55.0
    
    # 6. ASSISTS
    ast_line = round(projected_total / 9.5, 1)
    ast_pred = "OVER" if projected_total > 220 else "UNDER"
    ast_conf = 55.0
    
    # 7. 1ER CUARTO
    q1_winner = winner
    q1_conf = round(max(50.0, win_conf - 5.0), 1)
    
    return {
        "market_1_winner": {"prediction": winner, "confidence": win_conf},
        "market_2_spread": {"line": f"{spread_team} {spread_line}", "confidence": spread_conf},
        "market_3_ou": {"line": ou_line, "prediction": ou_prediction, "confidence": ou_conf},
        "market_4_threes": {"line": threes_line, "prediction": threes_pred, "confidence": threes_conf},
        "market_5_rebounds": {"line": reb_line, "prediction": reb_pred, "confidence": reb_conf},
        "market_6_assists": {"line": ast_line, "prediction": ast_pred, "confidence": ast_conf},
        "market_7_q1": {"prediction": q1_winner, "confidence": q1_conf},
        "team_stats": {
            "away": {"ppg": away_stats.get("ppg", 0), "opp_ppg": away_stats.get("opp_ppg", 0), "record": f"{away_stats.get('wins', 0)}-{away_stats.get('losses', 0)}"},
            "home": {"ppg": home_stats.get("ppg", 0), "opp_ppg": home_stats.get("opp_ppg", 0), "record": f"{home_stats.get('wins', 0)}-{home_stats.get('losses', 0)}"}
        },
        "signals": signals
    }
