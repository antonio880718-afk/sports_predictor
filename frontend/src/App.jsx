import { useState, useEffect } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('HOY')
  const [activeSport, setActiveSport] = useState('MLB')
  const [activeSoccerLeague, setActiveSoccerLeague] = useState('all')
  const [games, setGames] = useState([])
  const [history, setHistory] = useState([])
  const [trainingLogs, setTrainingLogs] = useState([])
  const [loading, setLoading] = useState(false)

  const sports = ['MLB', 'NBA', 'NFL', 'SOCCER', 'LMB']
  const tabs = ['INICIO', 'HOY', 'AUDITORIA']
  const soccerLeagues = [
    {id: 'all', name: 'Todas las Ligas'},
    {id: 'eng.1', name: 'Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
    {id: 'eng.2', name: 'Championship 🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
    {id: 'esp.1', name: 'LaLiga 🇪🇸'},
    {id: 'esp.2', name: 'LaLiga 2 🇪🇸'},
    {id: 'ita.1', name: 'Serie A 🇮🇹'},
    {id: 'ita.2', name: 'Serie B 🇮🇹'},
    {id: 'fra.1', name: 'Ligue 1 🇫🇷'},
    {id: 'ger.1', name: 'Bundesliga 🇩🇪'},
    {id: 'por.1', name: 'Primeira Liga 🇵🇹'},
    {id: 'ned.1', name: 'Eredivisie 🇳🇱'},
    {id: 'mex.1', name: 'Liga MX 🇲🇽'},
    {id: 'arg.1', name: 'Primera Arg 🇦🇷'},
    {id: 'bra.1', name: 'Brasileirão 🇧🇷'},
    {id: 'col.1', name: 'Liga BetPlay 🇨🇴'},
    {id: 'usa.1', name: 'MLS 🇺🇸'}
  ]

  const [performance, setPerformance] = useState(null)

  useEffect(() => {
    setLearningReport(null) // Limpiar el reporte al cambiar de pestaña
    if (activeTab === 'HOY') {
      fetchGames(activeSport, activeSoccerLeague)
    } else if (activeTab === 'AUDITORIA') {
      fetchHistory(activeSport)
      fetchTrainingLogs()
    } else if (activeTab === 'INICIO') {
      fetchPerformance(activeSport)
    }
  }, [activeSport, activeTab, activeSoccerLeague])

  const fetchPerformance = async (sport) => {
    setLoading(true)
    try {
      const sportId = sport === 'LMB' ? 23 : 1
      const response = await fetch(`https://sports-predictor-y4mq.onrender.com/api/ai/performance?sport_id=${sportId}`)
      const data = await response.json()
      setPerformance(data)
    } catch (error) {
      setPerformance(null)
    }
    setLoading(false)
  }

  const [auditDate, setAuditDate] = useState('2026-07-10')
  const [apiMessage, setApiMessage] = useState("")

  const fetchGames = async (sport, league = "all") => {
    setLoading(true)
    try {
      let url = `https://sports-predictor-y4mq.onrender.com/api/${sport.toLowerCase()}/today`
      if (sport === 'SOCCER') {
          url += `?league=${league}`
      }
      const response = await fetch(url)
      const data = await response.json()
      setGames(data.games || [])
      setApiMessage(data.message || "")
    } catch (error) {
      console.error("Error fetching data:", error)
      setGames([])
      setApiMessage("Error de conexión con el servidor.")
    }
    setLoading(false)
  }

  const fetchHistory = async (sport, dateStr = auditDate) => {
    if (sport !== 'MLB' && sport !== 'LMB') {
      setHistory([])
      setApiMessage("")
      return
    }
    setLoading(true)
    try {
      const response = await fetch(`https://sports-predictor-y4mq.onrender.com/api/${sport.toLowerCase()}/history?date=${dateStr}`)
      const data = await response.json()
      setHistory(data.results || [])
      setApiMessage(data.message || "")
    } catch (error) {
      console.error("Error fetching history:", error)
      setHistory([])
    }
    setLoading(false)
  }

  const fetchTrainingLogs = async () => {
    try {
      const response = await fetch('https://sports-predictor-y4mq.onrender.com/api/ai/training_logs')
      const data = await response.json()
      setTrainingLogs(data.logs || [])
    } catch (error) {
      console.error("Error fetching logs:", error)
    }
  }

  useEffect(() => {
    if (activeTab === 'AUDITORIA') {
      fetchHistory(activeSport, auditDate)
    }
  }, [auditDate])

  const [learningMode, setLearningMode] = useState(false)
  const [learningReport, setLearningReport] = useState(null)
  const [showTicketModal, setShowTicketModal] = useState(false)

  // Chat & Manual Training State
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMessage, setChatMessage] = useState('')
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', content: 'Hola. Soy Deep Props Engine. Pregúntame sobre mis tácticas o predicciones matemáticas.' }
  ])
  const [manualTrain, setManualTrain] = useState({
    sport: 'SOCCER', away_xg: 1.0, home_xg: 1.5, away_possession: 45, home_possession: 55, winner: 'Home'
  })
  const [manualTrainStatus, setManualTrainStatus] = useState('')

  const sendChatMessage = async () => {
    if(!chatMessage.trim()) return
    const userMsg = { role: 'user', content: chatMessage }
    setChatHistory(prev => [...prev, userMsg])
    setChatMessage('')
    try {
      const response = await fetch('https://sports-predictor-y4mq.onrender.com/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            message: userMsg.content, 
            sport: activeSport,
            history: chatHistory,
            live_data: games
        })
      })
      const data = await response.json()
      setChatHistory(prev => [...prev, { role: 'ai', content: data.response }])
    } catch (e) {
      setChatHistory(prev => [...prev, { role: 'ai', content: "Error de conexión con mi cerebro." }])
    }
  }

  const submitManualTrain = async () => {
    setManualTrainStatus('Inyectando datos...')
    try {
      const response = await fetch('https://sports-predictor-y4mq.onrender.com/api/ai/train_manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sport: manualTrain.sport,
          away_xg: parseFloat(manualTrain.away_xg),
          home_xg: parseFloat(manualTrain.home_xg),
          away_possession: parseFloat(manualTrain.away_possession),
          home_possession: parseFloat(manualTrain.home_possession),
          winner: manualTrain.winner
        })
      })
      const data = await response.json()
      setManualTrainStatus(data.message || data.error)
      if(data.status === "COMPLETADO") {
        fetchTrainingLogs()
      }
    } catch (e) {
      setManualTrainStatus('Error al inyectar datos.')
    }
  }

  const triggerLearning = async () => {
    setLearningMode(true)
    setLearningReport(null)
    try {
      const qs = activeTab === 'AUDITORIA' ? `?date=${auditDate}` : ""
      const response = await fetch(`https://sports-predictor-y4mq.onrender.com/api/${activeSport.toLowerCase()}/learn${qs}`, { method: 'POST' })
      const data = await response.json()
      setLearningReport(data)
    } catch (error) {
      console.error("Error triggering learning:", error)
    }
    setLearningMode(false)
  }

  const topPicks = games
    .filter(g => g.winProbability && g.winProbability.confidence !== '--%')
    .filter(g => parseFloat(g.winProbability.confidence) >= 70.0)
    .sort((a, b) => parseFloat(b.winProbability.confidence) - parseFloat(a.winProbability.confidence))
    .slice(0, 4)

  const otherGames = games.filter(g => !topPicks.includes(g))

  const renderGameCard = (game, isTopPick) => (
    <div key={game.gamePk} className={`glass-card rounded-2xl p-6 relative overflow-hidden ${isTopPick ? 'border-2 border-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.3)]' : ''}`}>
      {isTopPick && (
         <div className="absolute top-0 right-0 bg-emerald-500 text-white text-xs font-black px-4 py-1 rounded-bl-xl shadow-lg z-10">
           TOP PICK
         </div>
      )}
      <div className="absolute -top-10 -right-10 w-32 h-32 bg-sky-500/20 rounded-full blur-3xl"></div>
      
      <div className="flex justify-between items-center mb-6 relative z-10">
        <div className="text-center w-5/12">
          <p className="text-sm text-slate-400 uppercase tracking-wider mb-1">Visitante</p>
          <h3 className="text-xl font-bold text-slate-100">{game.away.team}</h3>
        </div>
        <div className="text-center w-2/12">
          <span className="text-sm font-bold bg-slate-800 px-3 py-1 rounded-full text-slate-300 border border-slate-700">VS</span>
        </div>
        <div className="text-center w-5/12">
          <p className="text-sm text-slate-400 uppercase tracking-wider mb-1">Local</p>
          <h3 className="text-xl font-bold text-slate-100">{game.home.team}</h3>
        </div>
      </div>

      <div className="space-y-4 border-t border-slate-700/50 pt-4 relative z-10">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800">
            <p className="text-xs text-slate-400 mb-2">ABRIDOR PROBABLE</p>
            <p className="font-semibold text-sky-300 mb-2">{game.away.probablePitcher}</p>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Proyección K's:</span>
              <span className="font-bold text-white">{game.away.projectedStrikeouts}</span>
            </div>
          </div>
          
          <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800">
            <p className="text-xs text-slate-400 mb-2">ABRIDOR PROBABLE</p>
            <p className="font-semibold text-sky-300 mb-2">{game.home.probablePitcher}</p>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Proyección K's:</span>
              <span className="font-bold text-white">{game.home.projectedStrikeouts}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 bg-slate-900/50 border border-slate-700 p-4 rounded-xl flex justify-between items-center relative z-10">
        <div>
          <p className="text-xs text-amber-400 uppercase tracking-widest">Línea de Carreras (O/U: {game.overUnderRuns?.line})</p>
          <p className="text-lg font-bold text-white mt-1">Predicción: {game.overUnderRuns?.prediction}</p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-black text-amber-400">
            {game.overUnderRuns?.confidence}
          </p>
        </div>
      </div>

      <div className={`mt-3 p-4 rounded-xl flex justify-between items-center relative z-10 ${isTopPick ? 'bg-emerald-900/50 border border-emerald-400/50' : 'bg-emerald-900/30 border border-emerald-500/30'}`}>
        <div>
          <p className="text-xs text-emerald-400 uppercase tracking-widest">Ganador del Partido</p>
          <p className="text-lg font-bold text-white mt-1">Favorito: {game.winProbability.favorite}</p>
        </div>
        <div className="text-right">
          <p className="text-3xl font-black glow-text-green text-emerald-400">
            {game.winProbability.confidence}
          </p>
        </div>
      </div>
    </div>
  )

  const renderNflCard = (game, isTopPick) => {
    const { nflMarkets } = game;
    if (!nflMarkets) return null;

    return (
      <div key={game.gamePk} className={`glass-card rounded-2xl p-6 relative overflow-hidden ${isTopPick ? 'border-2 border-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.3)]' : ''}`}>
        {isTopPick && (
           <div className="absolute top-0 right-0 bg-emerald-500 text-white text-xs font-black px-4 py-1 rounded-bl-xl shadow-lg z-10">
             TOP PICK
           </div>
        )}
        <div className="absolute -top-10 -right-10 w-32 h-32 bg-indigo-500/20 rounded-full blur-3xl"></div>
        
        <div className="flex justify-between items-center mb-4 relative z-10">
          <div className="text-center w-5/12">
            <h3 className="text-lg font-bold text-slate-100 leading-tight">{game.away.team}</h3>
          </div>
          <div className="text-center w-2/12">
            <span className="text-sm font-bold bg-slate-800 px-3 py-1 rounded-full text-slate-300 border border-slate-700">@</span>
          </div>
          <div className="text-center w-5/12">
            <h3 className="text-lg font-bold text-slate-100 leading-tight">{game.home.team}</h3>
          </div>
        </div>

        {/* 7 Mercados Grid */}
        <div className="grid grid-cols-2 gap-3 relative z-10">
          
          {/* 1. Winner */}
          <div className="col-span-2 bg-emerald-900/40 border border-emerald-500/40 p-3 rounded-lg flex justify-between items-center">
            <div>
              <p className="text-[10px] text-emerald-400 uppercase tracking-widest">1. Ganador (Moneyline)</p>
              <p className="text-base font-bold text-white">{nflMarkets.market_1_winner.prediction}</p>
            </div>
            <div className="text-xl font-black text-emerald-400">{nflMarkets.market_1_winner.confidence}%</div>
          </div>

          {/* 2. Spread */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-sky-400 uppercase tracking-widest">2. Línea (Spread)</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_2_spread.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_2_spread.confidence}%</p>
          </div>

          {/* 3. Totales */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-amber-400 uppercase tracking-widest">3. Totales ({nflMarkets.market_3_ou.line})</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_3_ou.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_3_ou.confidence}%</p>
          </div>

          {/* 4. Touchdowns */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-rose-400 uppercase tracking-widest">4. Touchdowns Totales</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_4_tds.prediction} {nflMarkets.market_4_tds.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_4_tds.confidence}%</p>
          </div>

          {/* 5. 1er en Anotar */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-purple-400 uppercase tracking-widest">5. 1ro en Anotar</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_7_first.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_7_first.confidence}%</p>
          </div>

          {/* 6. QB Props */}
          <div className="bg-indigo-900/30 border border-indigo-500/30 p-3 rounded-lg">
            <p className="text-[10px] text-indigo-400 uppercase tracking-widest">6. Yardas QB ({nflMarkets.market_5_qb.line})</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_5_qb.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_5_qb.confidence}%</p>
          </div>

          {/* 7. RB Props */}
          <div className="bg-orange-900/30 border border-orange-500/30 p-3 rounded-lg">
            <p className="text-[10px] text-orange-400 uppercase tracking-widest">7. Yardas RB ({nflMarkets.market_6_rb.line})</p>
            <p className="text-sm font-bold text-white">{nflMarkets.market_6_rb.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {nflMarkets.market_6_rb.confidence}%</p>
          </div>

        </div>
      </div>
    );
  }

  const renderSoccerCard = (game, isTopPick) => {
    const { soccerMarkets } = game;
    if (!soccerMarkets) return null;

    return (
      <div key={game.gamePk} className={`glass-card rounded-2xl p-6 relative overflow-hidden ${isTopPick ? 'border-2 border-emerald-500/50 shadow-[0_0_20px_rgba(16,185,129,0.3)]' : ''}`}>
        {isTopPick && (
           <div className="absolute top-0 right-0 bg-emerald-500 text-white text-xs font-black px-4 py-1 rounded-bl-xl shadow-lg z-10">
             TOP PICK
           </div>
        )}
        <div className="absolute -top-10 -right-10 w-32 h-32 bg-purple-500/20 rounded-full blur-3xl"></div>
        
        <div className="flex justify-between items-center mb-4 relative z-10">
          <div className="text-center w-5/12">
            <h3 className="text-lg font-bold text-slate-100 leading-tight">{game.home.team}</h3>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Local</p>
          </div>
          <div className="text-center w-2/12">
            <span className="text-sm font-bold bg-slate-800 px-3 py-1 rounded-full text-slate-300 border border-slate-700">VS</span>
          </div>
          <div className="text-center w-5/12">
            <h3 className="text-lg font-bold text-slate-100 leading-tight">{game.away.team}</h3>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest mt-1">Visitante</p>
          </div>
        </div>

        {/* 7 Mercados Grid */}
        <div className="grid grid-cols-2 gap-3 relative z-10 mt-4">
          
          {/* 1. Winner */}
          <div className="col-span-2 bg-emerald-900/40 border border-emerald-500/40 p-3 rounded-lg flex flex-col justify-center">
            <div className="flex justify-between items-center">
              <div>
                <p className="text-[10px] text-emerald-400 uppercase tracking-widest">1. Ganador (1X2)</p>
                <p className="text-base font-bold text-white">{soccerMarkets.market_1_winner.prediction}</p>
              </div>
              <div className="text-xl font-black text-emerald-400">{soccerMarkets.market_1_winner.confidence}%</div>
            </div>
            {soccerMarkets.tactical_report && (
              <div className="mt-3 pt-3 border-t border-emerald-500/20">
                <p className="text-xs text-emerald-300/80 leading-relaxed italic">
                  <span className="font-bold text-emerald-400">🤖 Análisis IA: </span>
                  {soccerMarkets.tactical_report}
                </p>
              </div>
            )}
          </div>

          {/* 2. Doble Op */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-sky-400 uppercase tracking-widest">2. Doble Oportunidad</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_2_dc.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_2_dc.confidence}%</p>
          </div>

          {/* 3. Doble Op HT */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-cyan-400 uppercase tracking-widest">3. Doble Op. (Mitad)</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_3_dc_ht.prediction}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_3_dc_ht.confidence}%</p>
          </div>

          {/* 4. Corners */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-rose-400 uppercase tracking-widest">4. Total Córners</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_4_corners.prediction} {soccerMarkets.market_4_corners.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_4_corners.confidence}%</p>
          </div>

          {/* 5. Offsides */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-purple-400 uppercase tracking-widest">5. Fueras de Juego</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_5_offsides.prediction} {soccerMarkets.market_5_offsides.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_5_offsides.confidence}%</p>
          </div>

          {/* 6. Tarjetas */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-amber-400 uppercase tracking-widest">6. Total Tarjetas</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_6_cards.prediction} {soccerMarkets.market_6_cards.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_6_cards.confidence}%</p>
          </div>

          {/* 7. Goles */}
          <div className="bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-orange-400 uppercase tracking-widest">7. Total Goles</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_7_goals.prediction} {soccerMarkets.market_7_goals.line}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_7_goals.confidence}%</p>
          </div>

          {/* 8. Ambos Marcan */}
          <div className="col-span-2 lg:col-span-1 bg-slate-800/60 border border-slate-700 p-3 rounded-lg">
            <p className="text-[10px] text-pink-400 uppercase tracking-widest">8. Ambos Marcan (BTTS)</p>
            <p className="text-sm font-bold text-white">{soccerMarkets.market_8_btts?.prediction || "SÍ"}</p>
            <p className="text-xs text-slate-400 mt-1">Conf: {soccerMarkets.market_8_btts?.confidence || "75.0"}%</p>
          </div>

        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen text-white p-6 font-sans">
      <header className="max-w-7xl mx-auto mb-10 flex flex-col md:flex-row justify-between items-center glass-panel p-6 rounded-2xl">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight glow-text text-sky-400">
            DEEP PROPS <span className="text-white">ENGINE</span>
          </h1>
          <p className="text-slate-400 mt-2">Predicciones Granulares & Self-Correction IA</p>
        </div>
        
        <div className="flex flex-col gap-4 mt-6 md:mt-0 items-end">
          {/* Sports Selector */}
          <div className="flex gap-2 bg-slate-900/50 p-2 rounded-xl border border-slate-700/50">
            {sports.map(s => (
              <button 
                key={s}
                onClick={() => setActiveSport(s)}
                className={`px-4 py-1.5 rounded-lg font-semibold transition-all duration-300 text-sm ${
                  activeSport === s 
                  ? 'bg-sky-500 text-white shadow-[0_0_15px_rgba(14,165,233,0.6)]' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {s}
              </button>
            ))}
          </div>

          {/* Main Tabs (Today vs History) */}
          <div className="flex gap-2">
            {tabs.map(t => (
              <button 
                key={t}
                onClick={() => setActiveTab(t)}
                className={`px-6 py-2 rounded-lg font-bold tracking-wider transition-all duration-300 ${
                  activeTab === t 
                  ? 'bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.6)]' 
                  : 'bg-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                {t === 'INICIO' ? 'INICIO' : t === 'HOY' ? 'PARTIDOS DE HOY' : 'AUDITORÍA (BACKTESTING)'}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center mb-6 border-b border-slate-700 pb-4 gap-4">
          <h2 className="text-2xl font-bold text-slate-200">
            {activeTab === 'INICIO' ? `Rendimiento Histórico IA - ${activeSport}` : 
             activeTab === 'HOY' ? `Cartelera en Vivo - ${activeSport}` : 
             `Comprobación de Efectividad - ${activeSport}`}
          </h2>
          
          {activeTab === 'HOY' && (
            <div className="flex items-center gap-4">
              {activeSport === 'SOCCER' && (
                <div className="bg-slate-900/50 flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 text-sm">
                  <span className="text-slate-400">Liga:</span>
                  <select 
                    value={activeSoccerLeague}
                    onChange={(e) => setActiveSoccerLeague(e.target.value)}
                    className="bg-transparent text-sky-400 font-bold outline-none cursor-pointer"
                  >
                    {soccerLeagues.map(l => (
                      <option key={l.id} value={l.id} className="bg-slate-800 text-white">{l.name}</option>
                    ))}
                  </select>
                </div>
              )}
              <div className="bg-slate-900/50 px-4 py-2 rounded-lg border border-slate-700 text-sm font-medium text-sky-400">
                📅 Fecha Oficial: {new Date().toLocaleDateString('es-ES')}
              </div>
            </div>
          )}

          {activeTab === 'AUDITORIA' && (
            <div className="flex items-center gap-4">
              <div className="bg-slate-900/50 flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 text-sm">
                <span className="text-slate-400">📅 Fecha:</span>
                <input 
                  type="date" 
                  value={auditDate}
                  onChange={(e) => setAuditDate(e.target.value)}
                  className="bg-transparent text-sky-400 font-bold outline-none cursor-pointer"
                />
              </div>

              <button 
                onClick={triggerLearning}
                disabled={learningMode}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-bold shadow-[0_0_15px_rgba(79,70,229,0.5)] flex items-center gap-2 transition-all disabled:opacity-50"
              >
                {learningMode ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                    Entrenando...
                  </>
                ) : (
                  '🧠 Forzar Aprendizaje IA'
                )}
              </button>
            </div>
          )}
        </div>
        
        {/* Learning Report Banner */}
        {learningReport && activeTab === 'AUDITORIA' && (
          <div className="mb-8 p-6 bg-indigo-900/40 border border-indigo-500/50 rounded-xl glass-panel animate-fade-in">
            <h3 className="text-xl font-bold text-indigo-300 mb-4 flex items-center gap-2">
              ⚠️ Reporte de Auto-Corrección (Self-Correction Engine)
            </h3>
            <p className="text-slate-300 mb-4">{learningReport.message}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {learningReport.insights.map((insight, idx) => (
                <div key={idx} className="bg-slate-900/60 p-4 rounded-lg border border-slate-700">
                  <p className="text-sm text-slate-400 mb-2">PATRÓN DE FALLO DETECTADO:</p>
                  <p className="text-white font-medium mb-3">{insight.patternFound}</p>
                  <p className="text-xs text-indigo-400 uppercase tracking-wider mb-1">Acción Correctiva (Pesos):</p>
                  <p className="text-sm text-emerald-400 font-bold">{insight.actionTaken}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-sky-500"></div>
          </div>
        ) : activeTab === 'INICIO' ? (
          <div className="bg-slate-900/40 rounded-2xl border border-slate-700/50 p-8 shadow-2xl">
            {performance ? (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
                <div className="bg-slate-800/80 p-6 rounded-xl border border-slate-700">
                  <h3 className="text-slate-400 text-sm font-bold tracking-widest mb-2">PARTIDOS EVALUADOS</h3>
                  <p className="text-5xl font-extrabold text-white">{performance.total_games}</p>
                </div>
                <div className="bg-emerald-900/30 p-6 rounded-xl border border-emerald-500/30">
                  <h3 className="text-emerald-400 text-sm font-bold tracking-widest mb-2">ACIERTOS</h3>
                  <p className="text-5xl font-extrabold text-emerald-400">{performance.correct}</p>
                </div>
                <div className="bg-rose-900/30 p-6 rounded-xl border border-rose-500/30">
                  <h3 className="text-rose-400 text-sm font-bold tracking-widest mb-2">ERRORES</h3>
                  <p className="text-5xl font-extrabold text-rose-400">{performance.errors}</p>
                </div>
                <div className="bg-sky-900/30 p-6 rounded-xl border border-sky-500/30">
                  <h3 className="text-sky-400 text-sm font-bold tracking-widest mb-2">EFECTIVIDAD</h3>
                  <p className="text-5xl font-extrabold text-sky-400 glow-text">{performance.win_rate}%</p>
                </div>
              </div>
            ) : (
              <div className="text-center text-slate-500">No hay datos históricos de esta liga. Presiona Forzar Aprendizaje en la pestaña de Auditoría.</div>
            )}
          </div>
        ) : activeTab === 'HOY' ? (
          // Vista de Hoy
          games.length > 0 ? (
            <div className="flex flex-col gap-10">
              {topPicks.length > 0 && (
                <div>
                  <h2 className="text-2xl font-bold text-emerald-400 mb-6 flex items-center gap-3">
                    <span className="text-3xl">🎯</span>
                    TOP {topPicks.length} PICKS DEL DÍA (Seguridad &gt; 70%)
                    <button 
                      onClick={() => setShowTicketModal(true)}
                      className="ml-auto text-sm bg-yellow-500 hover:bg-yellow-400 text-black px-4 py-2 rounded-lg font-black shadow-[0_0_15px_rgba(234,179,8,0.4)] uppercase flex items-center gap-2 transition-all transform hover:scale-105"
                    >
                      🎫 Generar Boleto Parley
                    </button>
                  </h2>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {topPicks.map(game => activeSport === 'SOCCER' ? renderSoccerCard(game, true) : (activeSport === 'NFL' ? renderNflCard(game, true) : renderGameCard(game, true)))}
                  </div>
                </div>
              )}
              
              {otherGames.length > 0 && (
                <div>
                  {topPicks.length > 0 && <h2 className="text-2xl font-bold text-slate-300 mb-6">Resto de la Cartelera</h2>}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {otherGames.map(game => activeSport === 'SOCCER' ? renderSoccerCard(game, false) : (activeSport === 'NFL' ? renderNflCard(game, false) : renderGameCard(game, false)))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="glass-panel p-10 text-center rounded-2xl flex flex-col items-center">
              <h3 className="text-2xl text-slate-300 mb-2">Cartelera Vacía</h3>
              <p className="text-slate-500 mb-4">{apiMessage || `El motor para ${activeSport} se está configurando.`}</p>
              <p className="text-sm text-sky-400/80 bg-sky-900/20 px-4 py-2 rounded-lg border border-sky-800/30">
                INFO: Si no ves partidos de Béisbol hoy, es porque la MLB no programó juegos para la fecha oficial (Posible All-Star break, lluvia extrema, o temporada baja).
              </p>
            </div>
          )
        ) : (
          // Vista de Auditoría (History / Logs)
          <div className="flex flex-col gap-8">
            {history.length > 0 ? (
              <div className="glass-panel rounded-2xl overflow-hidden">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-900/80 border-b border-slate-700">
                      <th className="p-4 text-slate-300 font-semibold">Partido</th>
                      <th className="p-4 text-slate-300 font-semibold text-center">Marcador Real</th>
                      <th className="p-4 text-slate-300 font-semibold">Predicción IA (Ganador)</th>
                      <th className="p-4 text-slate-300 font-semibold">Predicción IA (Total)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {history.map((h, idx) => {
                      const ouData = h.ouPrediction || h.goalsPrediction;
                      return (
                        <tr key={idx} className="hover:bg-slate-800/50 transition-colors">
                          <td className="p-4">
                            <p className="font-medium text-white">{h.matchup}</p>
                            {h.pitchers && <p className="text-xs text-slate-400 mt-1">Abridores: {h.pitchers}</p>}
                            {h.props && <p className="text-xs text-sky-400 mt-1">Jugador a seguir: {h.props.topHitter} (Prob Hit: {h.props.hitProb} | K's Proy: {h.props.kProj})</p>}
                            {h.league && <p className="text-xs text-slate-400 mt-1">Liga: {h.league.toUpperCase()}</p>}
                          </td>
                          <td className="p-4 text-center font-bold text-lg">{h.realScore || h.totalGoals || h.totalPoints || "-"}</td>
                          
                          {/* Ganador */}
                          <td className="p-4">
                            <div className="flex items-center gap-3">
                              <div>
                                <p className="font-bold">{h.aiPrediction.winner}</p>
                                <p className="text-xs text-slate-400">
                                  Confianza: {h.aiPrediction.confidence}% 
                                  {h.aiPrediction.isSniper && <span className="text-emerald-400 ml-1">(FRANCOTIRADOR)</span>}
                                </p>
                              </div>
                              {h.aiPrediction.hit ? (
                                <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 px-3 py-1 rounded-full text-xs font-bold shadow-[0_0_10px_rgba(16,185,129,0.3)]">
                                  ACIERTO
                                </span>
                              ) : (
                                <span className="bg-red-500/20 text-red-400 border border-red-500/50 px-3 py-1 rounded-full text-xs font-bold">
                                  FALLO
                                </span>
                              )}
                            </div>
                          </td>

                          {/* Over Under / Totales */}
                          <td className="p-4">
                            {ouData ? (
                              <div className="flex items-center gap-3">
                                <div>
                                  <p className="font-bold">{ouData.predicted}</p>
                                  <p className="text-xs text-slate-400">Línea: {ouData.line}</p>
                                </div>
                                {ouData.hit ? (
                                  <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/50 px-3 py-1 rounded-full text-xs font-bold shadow-[0_0_10px_rgba(16,185,129,0.3)]">
                                    ACIERTO
                                  </span>
                                ) : (
                                  <span className="bg-red-500/20 text-red-400 border border-red-500/50 px-3 py-1 rounded-full text-xs font-bold">
                                    FALLO
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="glass-panel p-10 text-center rounded-2xl">
                <h3 className="text-2xl text-slate-300 mb-2">Sin datos de auditoría</h3>
                <p className="text-slate-500">Aún no hay historial procesado para {activeSport} en la fecha {auditDate}.</p>
              </div>
            )}

            {activeSport === 'SOCCER' && (
              <>
                <div className="flex flex-col gap-6 mt-4">
                  <h3 className="text-2xl font-bold text-slate-300">Bitácora de Auto-Entrenamiento</h3>
                  {trainingLogs.length > 0 ? (
                    <div className="space-y-4">
                      {trainingLogs.map(log => (
                        <div key={log.id} className="bg-slate-800/50 p-4 rounded-xl border-l-4 border-l-emerald-500">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-bold text-emerald-400">Sesión IA #{log.id}</span>
                            <span className="text-xs text-slate-400 bg-slate-900 px-3 py-1 rounded-full border border-slate-700">{log.timestamp}</span>
                          </div>
                          <p className="text-slate-300 text-sm leading-relaxed">{log.message}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="glass-panel p-10 text-center rounded-2xl">
                      <p className="text-slate-500">La bitácora de entrenamiento está vacía. El Cron Job se ejecutará a las 10:00 AM.</p>
                    </div>
                  )}
                </div>

                {/* MANUAL TRAINING LAB */}
                <div className="glass-panel p-6 rounded-2xl border border-sky-900/30">
                  <h3 className="text-xl font-bold text-sky-400 mb-4">Laboratorio Manual (Inyección de Datos)</h3>
                  <p className="text-sm text-slate-400 mb-4">Introduce datos históricos reales para re-entrenar la red neuronal matemática. Actualmente optimizado para Soccer.</p>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div>
                      <label className="text-xs text-slate-500">xG Visitante</label>
                      <input type="number" step="0.1" value={manualTrain.away_xg} onChange={e=>setManualTrain({...manualTrain, away_xg: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white" />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500">xG Local</label>
                      <input type="number" step="0.1" value={manualTrain.home_xg} onChange={e=>setManualTrain({...manualTrain, home_xg: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white" />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500">Posesión Vis. (%)</label>
                      <input type="number" value={manualTrain.away_possession} onChange={e=>setManualTrain({...manualTrain, away_possession: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white" />
                    </div>
                    <div>
                      <label className="text-xs text-slate-500">Posesión Loc. (%)</label>
                      <input type="number" value={manualTrain.home_possession} onChange={e=>setManualTrain({...manualTrain, home_possession: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white" />
                    </div>
                    <div className="col-span-2">
                      <label className="text-xs text-slate-500">Ganador Real</label>
                      <select value={manualTrain.winner} onChange={e=>setManualTrain({...manualTrain, winner: e.target.value})} className="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white">
                        <option value="Home">Local (Home)</option>
                        <option value="Away">Visitante (Away)</option>
                        <option value="Draw">Empate (Draw)</option>
                      </select>
                    </div>
                    <div className="col-span-2 flex items-end">
                      <button onClick={submitManualTrain} className="w-full bg-red-600 hover:bg-red-500 text-white font-bold p-2 rounded shadow-[0_0_15px_rgba(220,38,38,0.5)] transition-all">
                        Inyectar Conocimiento a la IA
                      </button>
                    </div>
                  </div>
                  {manualTrainStatus && <p className="text-xs text-emerald-400 mt-2">{manualTrainStatus}</p>}
                </div>
              </>
            )}
          </div>
        )}
      </main>

      {/* TICKET MODAL */}
      {showTicketModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex justify-center items-center z-50 p-4">
          <div className="bg-[#f4f1ea] text-black w-full max-w-sm rounded-md shadow-[0_0_50px_rgba(255,255,255,0.1)] relative flex flex-col font-mono animate-fade-in">
            {/* Cabecera del ticket */}
            <div className="p-6 border-b-2 border-dashed border-gray-400 text-center">
              <h2 className="text-3xl font-black tracking-tighter uppercase mb-1">DEEP PROPS</h2>
              <p className="text-xs text-gray-500 uppercase tracking-widest font-bold">Apuesta Segura Parley</p>
              <p className="text-xs font-bold mt-2 text-gray-600">{new Date().toLocaleString()}</p>
            </div>
            
            {/* Picks */}
            <div className="p-6 flex-1 space-y-5 border-b-2 border-dashed border-gray-400">
              {topPicks.map((pick, i) => (
                <div key={i} className="text-sm">
                  <div className="flex justify-between font-black text-lg leading-tight">
                    <span>{pick.winProbability.favorite}</span>
                    <span>{pick.winProbability.confidence}</span>
                  </div>
                  <div className="text-xs text-gray-600 mt-1 font-bold">
                    {pick.away.team} @ {pick.home.team}
                  </div>
                  <div className="text-[10px] text-gray-500 mt-1 uppercase tracking-wider">
                    {pick.away.probablePitcher} vs {pick.home.probablePitcher}
                  </div>
                </div>
              ))}
            </div>

            {/* Footer / Resumen */}
            <div className="p-6 bg-gray-200/50 rounded-b-md">
              <div className="flex justify-between text-lg font-black uppercase mb-4">
                <span>Total Picks:</span>
                <span>{topPicks.length}</span>
              </div>
              <div className="text-center">
                <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2 font-bold">Validación en Caja</p>
                {/* Simulated Barcode */}
                <div className="flex justify-center gap-[2px] h-12 mb-2 w-full px-4">
                  {[...Array(40)].map((_, i) => (
                    <div key={i} className="bg-black h-full" style={{ width: `${Math.random() * 4 + 1}px` }}></div>
                  ))}
                </div>
                <p className="text-xs tracking-[0.5em] font-bold">8472910482</p>
              </div>
              <button 
                onClick={() => setShowTicketModal(false)}
                className="mt-6 w-full bg-black text-white font-bold py-3 rounded hover:bg-gray-800 transition-colors uppercase tracking-widest text-sm"
              >
                Cerrar Boleto
              </button>
            </div>
          </div>
        </div>
      )}

      {/* CHAT WIDGET */}
      <div className="fixed bottom-6 right-6 z-50">
        {!chatOpen ? (
          <button onClick={() => setChatOpen(true)} className="bg-sky-600 hover:bg-sky-500 text-white p-4 rounded-full shadow-[0_0_20px_rgba(2,132,199,0.6)] transition-all">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
          </button>
        ) : (
          <div className="bg-slate-900 border border-sky-900/50 rounded-2xl w-80 sm:w-96 shadow-2xl overflow-hidden flex flex-col" style={{ height: '500px' }}>
            <div className="bg-sky-900/50 p-4 flex justify-between items-center border-b border-sky-800">
              <h4 className="text-white font-bold flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
                Chat IA Táctico
              </h4>
              <button onClick={() => setChatOpen(false)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${msg.role === 'user' ? 'bg-sky-600 text-white rounded-br-none' : 'bg-slate-800 text-slate-300 border border-slate-700 rounded-bl-none'}`}>
                    {msg.content}
                  </div>
                </div>
              ))}
            </div>
            <div className="p-3 border-t border-slate-800 bg-slate-900 flex gap-2">
              <input 
                type="text" 
                value={chatMessage} 
                onChange={(e) => setChatMessage(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                placeholder="Pregúntale a la IA..."
                className="flex-1 bg-slate-800 border border-slate-700 text-white text-sm rounded-full px-4 py-2 focus:outline-none focus:border-sky-500"
              />
              <button onClick={sendChatMessage} className="bg-sky-600 hover:bg-sky-500 text-white rounded-full p-2 w-10 h-10 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
