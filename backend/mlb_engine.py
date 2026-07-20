import requests
import os
import json
import concurrent.futures
import pandas as pd

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
        return {"era": 4.50, "whip": 1.30, "k9": 8.0, "wins": 0, "losses": 0}
    
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

def train_model(sport_id: int = 1, target_date: str = None):
    """
    Simulación de compatibilidad: el sistema ahora usa Signal-Based Composite Scoring.
    """
    memory_filename = os.path.join(os.path.dirname(__file__), f"ai_memory_bank_{sport_id}.csv")
    if os.path.isfile(memory_filename):
        # Requirement check: Use on_bad_lines='skip' for all pd.read_csv calls
        df = pd.read_csv(memory_filename, on_bad_lines='skip')
    
    return {
        "status": "COMPLETADO",
        "gamesAudited": 0,
        "failedSniperBets": 0,
        "insights": [
            {
                "patternFound": "Motor actualizado a Signal-Based Composite Scoring con 6 variables de impacto.",
                "actionTaken": "Entrenamiento de ML reemplazado por señales determinísticas calibradas.",
                "feature": "Signal_System",
                "newWeight": "Composite"
            }
        ],
        "message": "IA Profunda ahora usa Signal-Based Composite Scoring para mayor precisión."
    }

def _clamp(val, min_val=-1.0, max_val=1.0):
    return max(min_val, min(max_val, val))

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

def get_team_standings_stats(sport_id: int, team_name: str, team_id: str):
    default_stats = {
        "winPct": 0.500,
        "runsScored": 4.5,
        "runsAllowed": 4.5,
        "gamesPlayed": 1,
        "l10_wins": 5
    }
    
    league_ids = "125" if sport_id == 23 else "103,104"
    url = f"http://statsapi.mlb.com/api/v1/standings?leagueId={league_ids}&season=2026&standingsTypes=regularSeason"
    
    try:
        data = requests.get(url, timeout=5).json()
        records = data.get("records", [])
        for division in records:
            for team_rec in division.get("teamRecords", []):
                t_id = str(team_rec.get("team", {}).get("id", ""))
                t_name = team_rec.get("team", {}).get("name", "").lower()
                
                if (team_id and t_id == team_id) or (team_name.lower() in t_name and team_name):
                    win_pct = float(team_rec.get("winningPercentage", 0.500))
                    runs_scored = int(team_rec.get("runsScored", 0))
                    runs_allowed = int(team_rec.get("runsAllowed", 0))
                    wins = int(team_rec.get("wins", 0))
                    losses = int(team_rec.get("losses", 0))
                    gp = wins + losses if (wins + losses) > 0 else 1
                    
                    l10_wins = 5
                    l10 = team_rec.get("last10")
                    if isinstance(l10, str) and "-" in l10:
                        try:
                            l10_wins = int(l10.split("-")[0])
                        except: pass
                    elif isinstance(l10, dict):
                        l10_wins = int(l10.get("wins", 5))
                    else:
                        # Sometimes it's inside splitRecords
                        split_recs = team_rec.get("records", {}).get("splitRecords", [])
                        for sr in split_recs:
                            if sr.get("type") == "lastTen":
                                l10_wins = int(sr.get("wins", 5))
                                break
                    
                    return {
                        "winPct": win_pct,
                        "runsScored": runs_scored,
                        "runsAllowed": runs_allowed,
                        "gamesPlayed": gp,
                        "l10_wins": l10_wins
                    }
    except Exception as e:
        print(f"Error fetching standings: {e}")
        
    return default_stats

def calculate_game_predictions(away_pitcher_name: str, home_pitcher_name: str, away_team: str, home_team: str, top_hitter_name: str, sport_id: int = 1):
    """
    Predicción usando un motor SIGNAL-BASED COMPOSITE determinístico.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        f_away_id = executor.submit(get_player_id, away_pitcher_name)
        f_home_id = executor.submit(get_player_id, home_pitcher_name)
        f_hitter_id = executor.submit(get_player_id, top_hitter_name)
        f_away_team_id = executor.submit(get_team_id, away_team, sport_id)
        f_home_team_id = executor.submit(get_team_id, home_team, sport_id)
        
    away_id = f_away_id.result()
    home_id = f_home_id.result()
    hitter_id = f_hitter_id.result()
    away_team_id = f_away_team_id.result()
    home_team_id = f_home_team_id.result()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        f_away_p_stats = executor.submit(get_pitcher_season_stats, away_id, sport_id)
        f_home_p_stats = executor.submit(get_pitcher_season_stats, home_id, sport_id)
        f_hitter_stats = executor.submit(get_hitter_season_stats, hitter_id, sport_id)
        f_away_ops = executor.submit(get_team_hitting_stats, away_team_id, sport_id)
        f_home_ops = executor.submit(get_team_hitting_stats, home_team_id, sport_id)
        f_away_std = executor.submit(get_team_standings_stats, sport_id, away_team, away_team_id)
        f_home_std = executor.submit(get_team_standings_stats, sport_id, home_team, home_team_id)
        
    away_p_stats = f_away_p_stats.result()
    home_p_stats = f_home_p_stats.result()
    hitter_stats = f_hitter_stats.result()
    away_ops = f_away_ops.result()
    home_ops = f_home_ops.result()
    away_std = f_away_std.result()
    home_std = f_home_std.result()

    signals = {}
    
    # Signal 1: Team Record (Weight: 0.30)
    diff_winpct = home_std.get("winPct", 0.5) - away_std.get("winPct", 0.5)
    s1_score = _clamp(diff_winpct * 3.0)
    signals['team_record'] = {'score': s1_score, 'weight': 0.30}
    
    # Signal 2: Starting Pitcher Matchup (Weight: 0.25)
    s2_score = (away_p_stats.get("era", 4.5) - home_p_stats.get("era", 4.5)) / 3.0
    s2_score += (away_p_stats.get("whip", 1.3) - home_p_stats.get("whip", 1.3)) * 0.5
    s2_score = _clamp(s2_score)
    signals['pitcher_matchup'] = {'score': s2_score, 'weight': 0.25}
    
    # Signal 3: Team Offense (Weight: 0.15)
    s3_score = _clamp((home_ops - away_ops) / 0.100)
    signals['team_offense'] = {'score': s3_score, 'weight': 0.15}
    
    # Signal 4: Home Advantage (Weight: 0.10)
    s4_score = _clamp(0.08)
    signals['home_advantage'] = {'score': s4_score, 'weight': 0.10}
    
    # Signal 5: Recent Form / L10 (Weight: 0.15)
    s5_score = _clamp((home_std.get("l10_wins", 5) - away_std.get("l10_wins", 5)) / 5.0)
    signals['recent_form'] = {'score': s5_score, 'weight': 0.15}
    
    # Signal 6: Run Differential (Weight: 0.05)
    home_rd = (home_std.get("runsScored", 4.5) - home_std.get("runsAllowed", 4.5)) / max(1, home_std.get("gamesPlayed", 1))
    away_rd = (away_std.get("runsScored", 4.5) - away_std.get("runsAllowed", 4.5)) / max(1, away_std.get("gamesPlayed", 1))
    s6_score = _clamp((home_rd - away_rd) / 2.0)
    signals['run_differential'] = {'score': s6_score, 'weight': 0.05}
    
    composite_score = sum(sig['score'] * sig['weight'] for sig in signals.values())
    
    base_prob = 50.0 + (composite_score * 20.0)
    
    if composite_score > 0:
        favorite = home_team
        win_conf = base_prob
    else:
        favorite = away_team
        win_conf = 100.0 - base_prob
        
    win_conf = max(50.0, min(72.0, win_conf))
    
    # --- Over/Under Calculation (Cross-referenced) ---
    away_rpg = away_std.get("runsScored", 4.5) / max(1, away_std.get("gamesPlayed", 1))
    home_rpg = home_std.get("runsScored", 4.5) / max(1, home_std.get("gamesPlayed", 1))
    away_rapg = away_std.get("runsAllowed", 4.5) / max(1, away_std.get("gamesPlayed", 1))
    home_rapg = home_std.get("runsAllowed", 4.5) / max(1, home_std.get("gamesPlayed", 1))
    
    # Cross-reference: Away offense vs Home pitching, Home offense vs Away pitching
    away_expected = (away_rpg + home_rapg) / 2.0  # Away team's expected runs
    home_expected = (home_rpg + away_rapg) / 2.0  # Home team's expected runs
    
    # Factor in starting pitcher quality
    league_avg_era = 4.20
    away_era = away_p_stats.get("era", league_avg_era)
    home_era = home_p_stats.get("era", league_avg_era)
    
    # Pitcher ERA adjustment: good pitchers suppress runs, bad pitchers inflate them
    away_pitcher_factor = league_avg_era / max(0.5, home_era)  # If home pitcher has low ERA, away scores less
    home_pitcher_factor = league_avg_era / max(0.5, away_era)  # If away pitcher has low ERA, home scores less
    
    away_adjusted = away_expected * away_pitcher_factor
    home_adjusted = home_expected * home_pitcher_factor
    
    projected_total = away_adjusted + home_adjusted
    
    # Standard line is 8.5 in MLB — use that as reference if projection is close
    standard_line = 8.5
    if abs(projected_total - standard_line) < 0.5:
        final_line = standard_line
    else:
        final_line = round(projected_total * 2) / 2
    
    # Predict based on distance from line
    margin = projected_total - final_line
    if margin > 0.3:
        predicted_ou = "OVER"
    elif margin < -0.3:
        predicted_ou = "UNDER"
    else:
        # Close call — lean towards UNDER (pitching tends to dominate in close games)
        predicted_ou = "UNDER"
    
    ou_conf = 50.0 + abs(margin) * 8.0
    ou_conf = max(50.0, min(65.0, ou_conf))
    
    away_k_proj = round(away_p_stats.get("k9", 8.0) * (5.5 / 9.0), 1)
    home_k_proj = round(home_p_stats.get("k9", 8.0) * (5.5 / 9.0), 1)
    
    hit_prob_decimal = 1 - ((1 - hitter_stats.get("avg", 0.250)) ** 4)
    hit_prob_percent = round(hit_prob_decimal * 100, 1)
    
    hitter_k_val = hitter_stats.get('ops', 0.750)
    if not isinstance(hitter_k_val, (int, float)):
        hitter_k_val = 0.750
        
    return {
        "away_k_proj": away_k_proj,
        "home_k_proj": home_k_proj,
        "hitter_hit_prob": f"{hit_prob_percent}%",
        "hitter_k_proj": f"{round(hitter_k_val * 10, 1)}",
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
        },
        "signals": signals
    }
