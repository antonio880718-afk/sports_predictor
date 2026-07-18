import requests
import os
import json
def get_player_id(full_name: str) -> str:
    """Busca el ID oficial de la MLB para un jugador dado su nombre completo."""
    if full_name == "TBD" or not full_name:
        return None
    url = f"http://statsapi.mlb.com/api/v1/people/search?names={full_name}"
    try:
        data = requests.get(url, timeout=5).json()
        if data.get("people") and len(data["people"]) > 0:
            return data["people"][0]["id"]
    except Exception as e:
        print(f"Error fetching ID for {full_name}: {e}")
    return None

def get_pitcher_season_stats(person_id: str, sport_id: int = 1):
    """Obtiene las estadísticas oficiales de pitcheo de la temporada actual (2026)."""
    if not person_id:
        return {"era": 4.50, "whip": 1.30, "k9": 8.0, "wins": 0, "losses": 0} # Promedios de liga
    
    url = f"http://statsapi.mlb.com/api/v1/people/{person_id}/stats?stats=season&group=pitching&sportId={sport_id}"
    try:
        data = requests.get(url, timeout=5).json()
        if "stats" in data and len(data["stats"]) > 0 and len(data["stats"][0]["splits"]) > 0:
            stats = data["stats"][0]["splits"][0]["stat"]
            return {
                "era": float(stats.get("era", 4.50) or 4.50),
                "whip": float(stats.get("whip", 1.30) or 1.30),
                "k9": float(stats.get("strikeoutsPer9Inn", 8.0) or 8.0),
                "wins": int(stats.get("wins", 0) or 0),
                "losses": int(stats.get("losses", 0) or 0)
            }
    except Exception as e:
        print(f"Error fetching pitching stats for {person_id}: {e}")
        
    return {"era": 4.50, "whip": 1.30, "k9": 8.0, "wins": 0, "losses": 0}

def get_hitter_season_stats(person_id: str, sport_id: int = 1):
    """Obtiene las estadísticas oficiales de bateo de la temporada actual."""
    if not person_id:
        return {"avg": 0.250, "ops": 0.750, "hr": 0}
        
    url = f"http://statsapi.mlb.com/api/v1/people/{person_id}/stats?stats=season&group=hitting&sportId={sport_id}"
    try:
        data = requests.get(url, timeout=5).json()
        if "stats" in data and len(data["stats"]) > 0 and len(data["stats"][0]["splits"]) > 0:
            stats = data["stats"][0]["splits"][0]["stat"]
            return {
                "avg": float(stats.get("avg", 0.250) or 0.250),
                "ops": float(stats.get("ops", 0.750) or 0.750),
                "hr": int(stats.get("homeRuns", 0) or 0)
            }
    except Exception as e:
        pass
        
    return {"avg": 0.250, "ops": 0.750, "hr": 0}

import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'mlb_ai_model.pkl')

import concurrent.futures

def get_team_hitting_stats(team_id: str, sport_id: int):
    """Obtiene las estadísticas ofensivas colectivas de un equipo."""
    if not team_id:
        return 0.750
    url = f"http://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&sportId={sport_id}"
    try:
        data = requests.get(url, timeout=5).json()
        if "stats" in data and len(data["stats"]) > 0 and len(data["stats"][0]["splits"]) > 0:
            return float(data["stats"][0]["splits"][0]["stat"].get("ops", 0.750) or 0.750)
    except:
        pass
    return 0.750

def get_game_features(game, sport_id):
    try:
        away_score = game['teams']['away'].get('score', 0)
        home_score = game['teams']['home'].get('score', 0)
        if away_score == 0 and home_score == 0: return None
        
        away_pitcher = game['teams']['away'].get('probablePitcher', {})
        home_pitcher = game['teams']['home'].get('probablePitcher', {})
        
        if not away_pitcher or not home_pitcher: return None
        
        away_id = str(away_pitcher.get('id', ''))
        home_id = str(home_pitcher.get('id', ''))
        
        if not away_id or not home_id: return None
        
        away_stats = get_pitcher_season_stats(away_id, sport_id=sport_id)
        home_stats = get_pitcher_season_stats(home_id, sport_id=sport_id)
        
        away_team_id = str(game['teams']['away']['team']['id'])
        home_team_id = str(game['teams']['home']['team']['id'])
        
        away_ops = get_team_hitting_stats(away_team_id, sport_id)
        home_ops = get_team_hitting_stats(home_team_id, sport_id)
        
        features = [
            away_stats["era"], away_stats["whip"], 
            home_stats["era"], home_stats["whip"],
            away_ops, home_ops,
            1.0
        ]
        
        y = 1 if home_score > away_score else 0
        return features, y
    except Exception as e:
        return None

def train_model(sport_id: int = 1, target_date: str = None):
    """
    Ejecuta el entrenamiento de Gradient Boosting con Memoria Permanente.
    Guarda cada experiencia en un CSV para aprender de todo el historial.
    """
    import datetime
    import csv
    
    dates_to_train = []
    if target_date:
        dates_to_train = [target_date]
    else:
        # Si no hay fecha, pre-poblamos desde el 01/07/2026 hasta ayer
        start_date = datetime.date(2026, 7, 1)
        end_date = datetime.date.today() - datetime.timedelta(days=1)
        current_date = start_date
        while current_date <= end_date:
            dates_to_train.append(current_date.strftime("%Y-%m-%d"))
            current_date += datetime.timedelta(days=1)
            
    league_id = 125 if sport_id == 23 else 103
    league_param = f"&leagueId={league_id}" if sport_id == 23 else ""
    
    all_games = []
    for d in dates_to_train:
        url = f"http://statsapi.mlb.com/api/v1/schedule/games/?sportId={sport_id}{league_param}&date={d}&hydrate=probablePitcher(note)"
        try:
            res = requests.get(url, timeout=5).json()
            if not res.get('dates'): continue
            games = res['dates'][0].get('games', [])
            all_games.extend(games)
        except:
            pass

    # Archivo de memoria
    memory_filename = os.path.join(os.path.dirname(__file__), f"ai_memory_bank_{sport_id}.csv")
    
    # 1. Extraer características nuevas (solo si no las hemos extraído ya, pero por simplicidad de código las agregamos al CSV)
    new_features = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_game_features, g, sport_id) for g in all_games]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                features, y = res
                new_features.append(features + [y])
                
    # 2. Guardar en Memoria Permanente
    if new_features:
        file_exists = os.path.isfile(memory_filename)
        with open(memory_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["away_era", "away_whip", "home_era", "home_whip", "away_ops", "home_ops", "home_adv", "y"])
            for row in new_features:
                writer.writerow(row)
                
    # 3. Leer TODA la memoria para entrenar
    X_train = []
    y_train = []
    if os.path.isfile(memory_filename):
        df = pd.read_csv(memory_filename, on_bad_lines='skip')
        df = df.drop_duplicates() # Evitar aprender el mismo juego dos veces
        df.to_csv(memory_filename, index=False) # Guardar limpio
        if len(df) > 0:
            X_train = df.iloc[:, :-1].values.tolist()
            y_train = df.iloc[:, -1].values.tolist()
            
    total_games = len(X_train)

    if total_games == 0:
        return {"error": "Insuficientes datos en la Memoria Permanente para entrenar."}

    # Crear y entrenar el modelo (Gradient Boosting - Auto Corrección)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_train, y_train) * 100
    
    model_filename = "lmb_ai_model.pkl" if sport_id == 23 else "mlb_ai_model.pkl"
    target_model_file = os.path.join(os.path.dirname(__file__), model_filename)
    joblib.dump(clf, target_model_file)
    
    importances = clf.feature_importances_
    
    return {
        "status": "COMPLETADO",
        "gamesAudited": total_games,
        "failedSniperBets": int((1 - (accuracy/100)) * total_games),
        "insights": [
            {
                "patternFound": f"Motor GBM (200 ciclos de corrección) alcanzó {round(accuracy, 1)}% de precisión en {total_games} juegos.",
                "actionTaken": f"Cerebro corregido guardado en {model_filename}.",
                "feature": "GradientBoosting_ErrorCorrection",
                "newWeight": "N/A"
            },
            {
                "patternFound": "Análisis de Importancia con Ofensiva de Equipo completado.",
                "actionTaken": f"ERA Visitante: {round(importances[0]*100, 1)}% | OPS Visitante: {round(importances[4]*100, 1)}% | OPS Local: {round(importances[5]*100, 1)}%.",
                "feature": "Feature_Importance",
                "newWeight": "Dinámico"
            }
        ],
        "message": f"IA Profunda ({model_filename}) re-entrenada con Gradient Boosting. Ha aprendido de sus errores."
    }

def calculate_game_predictions(away_pitcher_name: str, home_pitcher_name: str, away_team: str, home_team: str, top_hitter_name: str, sport_id: int = 1):
    """
    Predicción usando Inteligencia Artificial (Random Forest).
    """
    away_id = get_player_id(away_pitcher_name)
    home_id = get_player_id(home_pitcher_name)
    hitter_id = get_player_id(top_hitter_name)
    
    away_stats = get_pitcher_season_stats(away_id, sport_id=sport_id)
    home_stats = get_pitcher_season_stats(home_id, sport_id=sport_id)
    hitter_stats = get_hitter_season_stats(hitter_id, sport_id=sport_id)
    
    model_filename = "lmb_ai_model.pkl" if sport_id == 23 else "mlb_ai_model.pkl"
    target_model_file = os.path.join(os.path.dirname(__file__), model_filename)

    # Si no existe el modelo, entrenarlo
    if not os.path.exists(target_model_file):
        train_model(sport_id)
        
    try:
        clf = joblib.load(target_model_file)
    except:
        train_model(sport_id)
        clf = joblib.load(target_model_file)
        
    def get_team_id(team_name: str, sport_id: int):
        url = f"http://statsapi.mlb.com/api/v1/teams?sportId={sport_id}"
        try:
            res = requests.get(url, timeout=5).json()
            for t in res.get("teams", []):
                if team_name.lower() in t["name"].lower():
                    return str(t["id"])
        except:
            pass
        return ""

    away_team_id = get_team_id(away_team, sport_id)
    home_team_id = get_team_id(home_team, sport_id)
    
    away_ops = get_team_hitting_stats(away_team_id, sport_id)
    home_ops = get_team_hitting_stats(home_team_id, sport_id)
    
    features = [[
        away_stats["era"], away_stats["whip"], 
        home_stats["era"], home_stats["whip"],
        away_ops, home_ops, 
        1.0 # home flag
    ]]
    
    # La IA predice la probabilidad basado en los patrones de los árboles!
    probs = clf.predict_proba(features)[0] # [Prob_Away_Wins, Prob_Home_Wins]
    away_prob = float(probs[0] * 100)
    home_prob = float(probs[1] * 100)
    
    if home_prob > away_prob:
        favorite = home_team
        win_conf = home_prob
    else:
        favorite = away_team
        win_conf = away_prob
        
    # Tope realista: en béisbol, nadie predice con más de 75% de confianza real
    win_conf = min(win_conf, 75.0)
    # Piso: si el modelo no sabe, al menos es un volado
    win_conf = max(win_conf, 51.0)
    
    combined_era = away_stats["era"] + home_stats["era"]
    calculated_line = combined_era
    final_line = round(calculated_line * 2) / 2
    predicted_ou = "OVER" if combined_era >= 8.5 else "UNDER"
    variance = abs(combined_era - 9.0)
    ou_conf = 50 + (variance * 3)
    ou_conf = min(ou_conf, 70.0)
    
    away_k_proj = round(away_stats["k9"] * (5.5 / 9.0), 1)
    home_k_proj = round(home_stats["k9"] * (5.5 / 9.0), 1)
    
    hit_prob_decimal = 1 - ((1 - hitter_stats["avg"]) ** 4)
    hit_prob_percent = round(hit_prob_decimal * 100, 1)

    return {
        "away_k_proj": away_k_proj,
        "home_k_proj": home_k_proj,
        "hitter_hit_prob": f"{hit_prob_percent}%",
        "hitter_k_proj": f"{round(hitter_stats.get('ops', 0.750) * 10, 1)}",
        "winProbability": {
            "favorite": favorite,
            "confidence": f"{round(win_conf, 1)}%",
            "raw_conf": win_conf
        },
        "overUnder": {
            "line": max(6.5, min(12.5, final_line)),
            "prediction": predicted_ou,
            "confidence": f"{round(ou_conf, 1)}%",
            "raw_conf": ou_conf
        }
    }
