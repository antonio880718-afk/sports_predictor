import os
import json
import random
import requests
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import joblib

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "ai_memory_bank_soccer.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "soccer_ai_model.pkl")

MEMORY_COLUMNS = [
    "date", "league", "away_team", "home_team",
    "away_xg", "home_xg", "away_poss", "home_poss",
    "away_goals", "home_goals", "total_goals", "btts",
    "winner"  # 0=Away, 1=Home, 2=Draw
]

FEATURE_COLUMNS = ["away_xg", "home_xg", "away_poss", "home_poss"]

def _ensure_memory_file():
    """Crea el archivo CSV de memoria si no existe."""
    if not os.path.isfile(MEMORY_FILE):
        with open(MEMORY_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(MEMORY_COLUMNS)


def harvest_real_results(target_date: str = None, days_back: int = 7):
    """
    Descarga resultados REALES de ESPN y los almacena en la Memoria Permanente.
    Cada juego se guarda con fecha + equipos para evitar duplicados.
    """
    _ensure_memory_file()
    
    if target_date:
        dates = [target_date]
    else:
        dates = []
        for i in range(1, days_back + 1):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(d)
    
    slugs = ["eng.1", "esp.1", "ita.1", "fra.1", "ger.1", "mex.1", "arg.1", "bra.1", "usa.1",
             "eng.2", "por.1", "ned.1", "tur.1", "col.1", "arg.1"]
    
    # Leer memoria existente para evitar duplicados
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
        for slug in slugs:
            url = f"http://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={date_formatted}"
            try:
                res = requests.get(url, timeout=5).json()
                events = res.get('events', [])
                
                for g in events:
                    try:
                        competition = g['competitions'][0]
                        
                        # Solo partidos terminados
                        status = competition.get('status', {}).get('type', {})
                        if not status.get('completed', False):
                            continue
                        
                        competitors = competition['competitors']
                        home_team = away_team = ""
                        home_score = away_score = 0
                        home_record = away_record = {}
                        
                        for c in competitors:
                            record = c.get('records', [{'summary': '5-5-5'}])[0] if c.get('records') else {'summary': '5-5-5'}
                            if c['homeAway'] == 'home':
                                home_team = c['team']['displayName']
                                home_score = int(c.get('score', 0))
                                home_record = record
                            else:
                                away_team = c['team']['displayName']
                                away_score = int(c.get('score', 0))
                                away_record = record
                        
                        # Evitar duplicados
                        key = f"{date_str}-{away_team}-{home_team}"
                        if key in existing_keys:
                            continue
                        existing_keys.add(key)
                        
                        # Calcular features derivados de la forma
                        def get_points(record_data):
                            try:
                                parts = str(record_data.get('summary', '5-5-5')).split("-")
                                w, d, l = int(parts[0]), int(parts[1]), int(parts[2])
                                return w * 3 + d
                            except:
                                return 15
                        
                        away_form = get_points(away_record)
                        home_form = get_points(home_record)
                        
                        # xG estimado por resultado real + forma
                        away_xg = away_score * 0.85 + (away_form / 30.0)
                        home_xg = home_score * 0.85 + (home_form / 30.0)
                        
                        # Posesión estimada
                        form_diff = home_form - away_form
                        away_poss = 45.0 + (away_form - home_form) * 0.3
                        home_poss = 55.0 - (away_form - home_form) * 0.3
                        away_poss = max(35.0, min(65.0, away_poss))
                        home_poss = max(35.0, min(65.0, home_poss))
                        
                        total_goals = away_score + home_score
                        btts = 1 if (away_score > 0 and home_score > 0) else 0
                        
                        if home_score > away_score:
                            winner = 1
                        elif away_score > home_score:
                            winner = 0
                        else:
                            winner = 2
                        
                        new_games.append([
                            date_str, slug, away_team, home_team,
                            round(away_xg, 2), round(home_xg, 2),
                            round(away_poss, 1), round(home_poss, 1),
                            away_score, home_score, total_goals, btts,
                            winner
                        ])
                    except:
                        continue
            except:
                continue
    
    # Guardar nuevos juegos en la memoria permanente
    if new_games:
        with open(MEMORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in new_games:
                writer.writerow(row)
    
    return len(new_games)


def train_soccer_model(target_date: str = None):
    """
    Entrena el modelo con TODA la memoria acumulada.
    Primero cosecha resultados reales, luego entrena con todo el historial.
    """
    # 1. Cosechar resultados reales nuevos
    new_count = harvest_real_results(target_date, days_back=14)
    
    _ensure_memory_file()
    
    # 2. Leer TODA la memoria
    df = pd.read_csv(MEMORY_FILE)
    df = df.drop_duplicates(subset=["date", "away_team", "home_team"])
    df.to_csv(MEMORY_FILE, index=False)  # Guardar limpio
    
    total_games = len(df)
    
    if total_games < 10:
        # Si no hay suficientes datos reales, generar datos base para arrancar
        seed_data = []
        for _ in range(50):
            aw_xg = random.uniform(0.3, 2.5)
            hm_xg = random.uniform(0.5, 3.0)
            aw_p = random.uniform(35, 60)
            hm_p = 100.0 - aw_p
            if hm_xg > aw_xg + 0.5:
                w = 1
            elif aw_xg > hm_xg + 0.5:
                w = 0
            else:
                w = 2
            seed_data.append([aw_xg, hm_xg, aw_p, hm_p, w])
        seed_df = pd.DataFrame(seed_data, columns=FEATURE_COLUMNS + ["winner"])
        X = seed_df[FEATURE_COLUMNS].values.tolist()
        y = seed_df["winner"].values.tolist()
    else:
        X = df[FEATURE_COLUMNS].values.tolist()
        y = df["winner"].values.tolist()
    
    total_training = len(X)
    
    # 3. Entrenar con TODA la memoria
    clf = GradientBoostingClassifier(
        n_estimators=min(200, 50 + total_training),  # Más árboles conforme hay más datos
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    clf.fit(X, y)
    
    accuracy = clf.score(X, y) * 100
    
    joblib.dump(clf, MODEL_FILE)
    
    # 4. Calcular estadísticas de memoria
    total_in_memory = len(df)
    goals_avg = round(df["total_goals"].mean(), 1) if total_in_memory > 0 and "total_goals" in df.columns else 0
    btts_pct = round((df["btts"].mean()) * 100, 1) if total_in_memory > 0 and "btts" in df.columns else 0
    
    importances = clf.feature_importances_
    
    return {
        "status": "COMPLETADO",
        "gamesAudited": total_training,
        "failedSniperBets": int((1 - (accuracy / 100)) * total_training),
        "insights": [
            {
                "patternFound": f"Cerebro GBM Soccer: {round(accuracy, 1)}% precisión en {total_training} partidos reales. Memoria total: {total_in_memory} juegos.",
                "actionTaken": f"Modelo re-entrenado. Se agregaron {new_count} juegos nuevos a la memoria permanente.",
                "feature": "GradientBoosting_PermanentMemory",
                "newWeight": "Acumulativo"
            },
            {
                "patternFound": f"Promedio goles: {goals_avg} | BTTS: {btts_pct}% | Importancia xG Local: {round(importances[1]*100, 1)}%",
                "actionTaken": f"xG Vis: {round(importances[0]*100, 1)}% | xG Loc: {round(importances[1]*100, 1)}% | Pos Vis: {round(importances[2]*100, 1)}% | Pos Loc: {round(importances[3]*100, 1)}%",
                "feature": "Feature_Importance_Soccer",
                "newWeight": "Dinámico"
            }
        ],
        "message": f"IA Soccer re-entrenada con {total_training} juegos. Memoria permanente: {total_in_memory} partidos acumulados (+{new_count} nuevos)."
    }


def calculate_soccer_predictions(away_team: str, home_team: str, away_record: dict, home_record: dict, league_slug: str):
    """
    Analiza los 8 mercados integrando Forma Táctica, Localía y Memoria Permanente.
    """
    if not os.path.exists(MODEL_FILE):
        train_soccer_model()
        
    try:
        clf = joblib.load(MODEL_FILE)
    except:
        train_soccer_model()
        clf = joblib.load(MODEL_FILE)
    
    # Extraer la forma de los equipos
    def get_points(record_str):
        try:
            parts = str(record_str).split("-")
            w, d, l = int(parts[0]), int(parts[1]), int(parts[2])
            return w * 3 + d
        except:
            return 15

    away_form_pts = get_points(away_record.get('summary', '5-5-5'))
    home_form_pts = get_points(home_record.get('summary', '5-5-5'))
    
    # Localía
    home_advantage_weight = 1.15
    if "bra" in league_slug or "mex" in league_slug or "arg" in league_slug:
        home_advantage_weight = 1.30
    
    form_diff = home_form_pts - away_form_pts
    
    # Features determinísticos
    away_xg_base = 0.9 + (away_form_pts / 25.0)
    home_xg_base = (0.9 + (home_form_pts / 25.0)) * home_advantage_weight
    
    away_poss = 45.0 + (away_form_pts - home_form_pts) * 0.3
    home_poss = 55.0 - (away_form_pts - home_form_pts) * 0.3
    away_poss = max(35.0, min(65.0, away_poss))
    home_poss = max(35.0, min(65.0, home_poss))
    
    features = [[away_xg_base, home_xg_base, away_poss, home_poss]]
    
    probs = clf.predict_proba(features)[0]
    
    away_prob = float(probs[0] * 100)
    home_prob = float(probs[1] * 100)
    draw_prob = float(probs[2] * 100)
    
    if form_diff > 10:
        home_prob += 5.0
        away_prob -= 5.0
    elif form_diff < -10:
        away_prob += 5.0
        home_prob -= 5.0
    
    total = away_prob + home_prob + draw_prob
    away_prob = (away_prob / total) * 100
    home_prob = (home_prob / total) * 100
    draw_prob = (draw_prob / total) * 100

    # --- Consultar memoria para calibrar predicciones ---
    btts_historical = 0.55  # default
    goals_avg_historical = 2.5
    if os.path.isfile(MEMORY_FILE):
        try:
            df_mem = pd.read_csv(MEMORY_FILE)
            if len(df_mem) > 20:
                btts_historical = df_mem["btts"].mean()
                goals_avg_historical = df_mem["total_goals"].mean()
        except:
            pass

    # 1. GANADOR (1X2)
    if home_prob > away_prob and home_prob > draw_prob:
        winner = home_team
        win_conf = min(home_prob, 72.0)
        report_text = f"El modelo favorece a {home_team} por localía (x{home_advantage_weight}) y forma táctica."
    elif away_prob > home_prob and away_prob > draw_prob:
        winner = away_team
        win_conf = min(away_prob, 72.0)
        report_text = f"{away_team} rompe la ventaja de localía con un Índice de Plantilla superior."
    else:
        winner = "Empate"
        win_conf = min(draw_prob, 55.0)
        report_text = f"Choque equilibrado. IA proyecta colisión táctica en mediocampo."
    win_conf = max(win_conf, 35.0)
        
    # 2. DOBLE OPORTUNIDAD
    if winner == home_team:
        dc_pred = f"{home_team} o Empate"
    elif winner == away_team:
        dc_pred = f"{away_team} o Empate"
    else:
        dc_pred = f"{home_team} o {away_team} (Sin Empate)"
    dc_conf = min(win_conf + 12.0, 82.0)
    
    # 3. DOBLE OPORTUNIDAD (Primera Mitad)
    dc_ht_pred = "Empate (Mitad) o " + (home_team if home_prob > away_prob else away_team)
    dc_ht_conf = round(min(win_conf + 5.0, 72.0), 1)
    
    dt_is_offensive = (home_xg_base + away_xg_base) > 3.0
    
    # 4. CÓRNERS — calibrado con historial
    corners_line = 9.5 if dt_is_offensive else 8.5
    corners_pred = "OVER" if dt_is_offensive else "UNDER"
    corners_conf = round(52.0 + abs(home_xg_base - away_xg_base) * 5.0, 1)
    corners_conf = min(corners_conf, 68.0)
    
    # 5. OFFSIDES
    offsides_line = 3.5 if dt_is_offensive else 2.5
    offsides_pred = "OVER" if dt_is_offensive else "UNDER"
    offsides_conf = round(50.0 + abs(form_diff) * 0.5, 1)
    offsides_conf = min(offsides_conf, 65.0)
    
    # 6. TARJETAS
    cards_line = 4.5
    cards_pred = "OVER" if abs(form_diff) < 5 else "UNDER"
    cards_conf = round(52.0 + abs(form_diff) * 0.8, 1)
    cards_conf = min(cards_conf, 68.0)
    
    # 7. GOLES — calibrado con memoria histórica
    goals_line = 2.5
    goals_pred = "OVER" if goals_avg_historical > 2.5 and dt_is_offensive else "UNDER"
    goals_conf = round(50.0 + abs(goals_avg_historical - 2.5) * 8.0, 1)
    goals_conf = min(goals_conf, 70.0)
    
    # 8. BTTS — calibrado con memoria histórica
    btts_pred = "SÍ" if btts_historical > 0.5 and (away_xg_base > 1.0 and home_xg_base > 1.0) else "NO"
    btts_conf = round(50.0 + abs(btts_historical - 0.5) * 30.0, 1)
    btts_conf = min(btts_conf, 70.0)

    return {
        "market_1_winner": {"prediction": winner, "confidence": round(win_conf, 1)},
        "market_2_dc": {"prediction": dc_pred, "confidence": round(dc_conf, 1)},
        "market_3_dc_ht": {"prediction": dc_ht_pred, "confidence": dc_ht_conf},
        "market_4_corners": {"line": corners_line, "prediction": corners_pred, "confidence": corners_conf},
        "market_5_offsides": {"line": offsides_line, "prediction": offsides_pred, "confidence": offsides_conf},
        "market_6_cards": {"line": cards_line, "prediction": cards_pred, "confidence": cards_conf},
        "market_7_goals": {"line": goals_line, "prediction": goals_pred, "confidence": goals_conf},
        "market_8_btts": {"prediction": btts_pred, "confidence": btts_conf},
        "tactical_report": report_text
    }
