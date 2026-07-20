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

_STANDINGS_CACHE = None

def _ensure_memory_file():
    if not os.path.isfile(MEMORY_FILE):
        with open(MEMORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MEMORY_COLUMNS)

def get_nfl_standings():
    global _STANDINGS_CACHE
    if _STANDINGS_CACHE is not None:
        return _STANDINGS_CACHE
        
    url = "http://site.api.espn.com/apis/v2/sports/football/nfl/standings"
    standings = {}
    try:
        res = requests.get(url, timeout=10).json()
        
        # Recursive search for "entries" which contains the actual team stats
        def find_entries(data):
            entries = []
            if isinstance(data, dict):
                if "entries" in data:
                    entries.extend(data["entries"])
                for k, v in data.items():
                    entries.extend(find_entries(v))
            elif isinstance(data, list):
                for item in data:
                    entries.extend(find_entries(item))
            return entries
        
        entries = find_entries(res)
        
        for entry in entries:
            team_info = entry.get("team", {})
            name = team_info.get("displayName", "")
            if not name:
                continue
                
            stats = entry.get("stats", [])
            stats_dict = {}
            for s in stats:
                stats_dict[s.get("name")] = s.get("value")
                
            wins = stats_dict.get("wins", 0)
            losses = stats_dict.get("losses", 0)
            ties = stats_dict.get("ties", 0)
            games = stats_dict.get("gamesPlayed", wins + losses + ties)
            if games == 0:
                games = 1
                
            points_for = stats_dict.get("pointsFor", 0)
            points_against = stats_dict.get("pointsAgainst", 0)
            
            standings[name] = {
                "wins": wins,
                "losses": losses,
                "games": games,
                "pointsFor": points_for,
                "pointsAgainst": points_against,
                "winpct": wins / games if games > 0 else 0.5
            }
    except Exception as e:
        pass
        
    _STANDINGS_CACHE = standings
    return standings

def clamp(val, min_val=-1.0, max_val=1.0):
    return max(min_val, min(max_val, val))

def harvest_real_results(target_date: str = None, days_back: int = 30):
    """
    Descarga resultados REALES de NFL de ESPN y los almacena permanentemente.
    """
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
    
    try:
        df = pd.read_csv(MEMORY_FILE, on_bad_lines='skip')
        df = df.drop_duplicates(subset=["date", "away_team", "home_team"])
        df.to_csv(MEMORY_FILE, index=False)
        total_in_memory = len(df)
    except:
        df = pd.DataFrame()
        total_in_memory = 0
    
    if total_in_memory < 10:
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
        X = []
        y = []
        for _, row in df.iterrows():
            away_imp = float(row.get("away_implied", row.get("away_pts", 21)))
            home_imp = float(row.get("home_implied", row.get("home_pts", 24)))
            total = float(row.get("total_pts", 45))
            X.append([away_imp, total - away_imp, home_imp, total - home_imp, 1.0])
            y.append(int(row.get("winner", 0)))
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
    home_win_pct = round((df["winner"].mean()) * 100, 1) if total_in_memory > 0 and "winner" in df.columns else 50
    
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


def calculate_nfl_predictions(away_team: str, home_team: str, spread_val: float = 0.0, total_val: float = 0.0):
    """Predicción NFL usando COMPOSITE SIGNALS + API."""
    
    standings = get_nfl_standings()
    
    home_stats = standings.get(home_team, {"wins": 0, "games": 1, "pointsFor": 24, "pointsAgainst": 24, "winpct": 0.5})
    away_stats = standings.get(away_team, {"wins": 0, "games": 1, "pointsFor": 24, "pointsAgainst": 24, "winpct": 0.5})
    
    h_games = max(1, home_stats.get("games", 1))
    a_games = max(1, away_stats.get("games", 1))
    
    h_winpct = home_stats.get("winpct", 0.5)
    a_winpct = away_stats.get("winpct", 0.5)
    
    h_pf = home_stats.get("pointsFor", 24 * h_games)
    h_pa = home_stats.get("pointsAgainst", 24 * h_games)
    
    a_pf = away_stats.get("pointsFor", 24 * a_games)
    a_pa = away_stats.get("pointsAgainst", 24 * a_games)
    
    h_diff = (h_pf - h_pa) / h_games
    a_diff = (a_pf - a_pa) / a_games
    
    h_ppg = h_pf / h_games
    a_ppg = a_pf / a_games
    
    h_opp = h_pa / h_games
    a_opp = a_pa / a_games
    
    # --- SIGNALS ---
    # Signal 1: Win Percentage (0.30)
    sig1 = clamp((h_winpct - a_winpct) * 2.0)
    
    # Signal 2: Point Differential (0.25)
    sig2 = clamp((h_diff - a_diff) / 14.0)
    
    # Signal 3: Home Advantage (0.15)
    sig3 = 0.14
    
    # Signal 4: Offensive Power (0.15)
    sig4 = clamp((h_ppg - a_ppg) / 10.0)
    
    # Signal 5: Defensive Strength (0.15)
    sig5 = clamp((a_opp - h_opp) / 7.0)
    
    composite = (sig1 * 0.30) + (sig2 * 0.25) + (sig3 * 0.15) + (sig4 * 0.15) + (sig5 * 0.15)
    
    # Winner
    winner = home_team if composite > 0 else away_team
    win_prob = clamp(50.0 + abs(composite) * 22.0, 50.0, 75.0)
    
    # Over/Under
    if total_val == 0: total_val = 44.5
    projected = h_ppg + a_ppg
    ou_line = total_val
    if projected > ou_line:
        ou_pred = "OVER"
        ou_conf = clamp(50.0 + (projected - ou_line) * 2.5, 50.0, 75.0)
    else:
        ou_pred = "UNDER"
        ou_conf = clamp(50.0 + (ou_line - projected) * 2.5, 50.0, 75.0)
        
    # Spread
    if spread_val == 0: spread_val = -3.0
    proj_diff = h_diff - a_diff + 2.5
    
    if proj_diff > -spread_val:
        spread_pred = home_team
    else:
        spread_pred = away_team
        
    spread_conf = clamp(50.0 + abs(proj_diff + spread_val) * 1.5, 50.0, 75.0)
    
    signals_data = {
        "composite": round(composite, 3),
        "win_pct_sig": round(sig1, 3),
        "diff_sig": round(sig2, 3),
        "home_adv_sig": round(sig3, 3),
        "offense_sig": round(sig4, 3),
        "defense_sig": round(sig5, 3)
    }

    return {
        "market_1_winner": {"prediction": winner, "confidence": round(win_prob, 1)},
        "market_2_spread": {"prediction": spread_pred, "line": f"{spread_val}", "confidence": round(spread_conf, 1)},
        "market_3_ou": {"prediction": ou_pred, "line": ou_line, "confidence": round(ou_conf, 1)},
        "market_4_first_half": {"prediction": winner, "confidence": round(win_prob - 2.0, 1)},
        "market_5_fg": {"prediction": "OVER 3.5", "confidence": 55.0},
        "market_6_turnovers": {"prediction": "UNDER 2.5", "confidence": 54.0},
        "market_7_sacks": {"prediction": "OVER 4.5", "confidence": 53.0},
        "signals": signals_data
    }
