import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { gamesApi, analysesApi, NextPrediction } from '../api/client'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, CheckCircle, XCircle, Target, Award, Sparkles } from 'lucide-react'

export default function Backtest() {
  const { t } = useTranslation()
  const [games, setGames] = useState<any[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [nTestDraws, setNTestDraws] = useState(20)
  const [topN, setTopN] = useState(10)
  const [nCombinations, setNCombinations] = useState(10)
  const [maxCommonMain, setMaxCommonMain] = useState(2)
  const [maxCommonBonus, setMaxCommonBonus] = useState(1)
  const [selectedModels, setSelectedModels] = useState<string[]>(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20', 'ANTI', 'ANTI2'])
  const [nextPrediction, setNextPrediction] = useState<NextPrediction | null>(null)
  const [loadingPrediction, setLoadingPrediction] = useState(false)

  useEffect(() => {
    loadGames()
  }, [])

  const loadGames = async () => {
    try {
      const response = await gamesApi.list()
      setGames(response.data)
      if (response.data.length > 0) {
        setSelectedGameId(response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load games:', error)
    }
  }

  const handleRunBacktest = async () => {
    if (!selectedGameId) return

    setLoading(true)
    try {
      const response = await analysesApi.run({
        game_id: selectedGameId,
        analysis_name: 'backtest_validation',
        params: {
          n_test_draws: nTestDraws,
          top_n: topN,
          n_combinations: nCombinations,
          max_common_main: maxCommonMain,
          max_common_bonus: maxCommonBonus,
          models: selectedModels
        },
      })
      console.log('Backtest response:', JSON.stringify(response.data, null, 2))
      setResult(response.data)
    } catch (error: any) {
      console.error('Backtest failed:', error)
      setResult({ error: error.response?.data?.detail || 'Backtest failed' })
    } finally {
      setLoading(false)
    }
  }

  const toggleModel = (model: string) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    )
  }

  const loadNextPrediction = async () => {
    if (!selectedGameId) return
    
    setLoadingPrediction(true)
    try {
      const response = await analysesApi.getNextPrediction(selectedGameId, nCombinations)
      setNextPrediction(response.data)
    } catch (error: any) {
      console.error('Failed to load prediction:', error)
      setNextPrediction(null)
    } finally {
      setLoadingPrediction(false)
    }
  }

  const getComparisonChartData = () => {
    if (!result?.results?.backtest?.models) return []
    
    return Object.entries(result.results.backtest.models).map(([name, data]: [string, any]) => ({
      model: name,
      avgHitRate: (data.statistics?.avg_hit_rate || 0) * 100,
      maxHitRate: (data.statistics?.max_hit_rate || 0) * 100,
      liftVsRandom: data.statistics?.lift_vs_random || 0,
      totalHits: data.statistics?.total_hits || 0
    }))
  }

  const renderModelResults = (modelName: string, modelData: any) => {
    if (!modelData || modelData.error) return null

    const stats = modelData.statistics || {}
    const predictions = modelData.predictions || []
    
    // Skip if no predictions
    if (predictions.length === 0) return null
    
    // Check if game has bonus numbers
    const hasBonus = predictions.some((p: any) => p.n_bonus_drawn > 0 || (p.predicted_bonus && p.predicted_bonus.length > 0))
    
    // Debug: log predictions for ANTI model
    if (modelName === 'ANTI' && predictions.length > 0) {
      console.log('ANTI predictions:', predictions[0])
    }

    return (
      <div key={modelName} className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          {modelName} - Résultats du Backtest
        </h3>

        {/* Statistics Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded">
            <p className="text-sm text-blue-700">Taux réussite (Principaux)</p>
            <p className="text-2xl font-bold text-blue-900">
              {((stats.avg_main_hit_rate || stats.avg_hit_rate || 0) * 100).toFixed(2)}%
            </p>
          </div>
          {hasBonus && (
            <div className="bg-cyan-50 p-4 rounded">
              <p className="text-sm text-cyan-700">Taux réussite (Bonus)</p>
              <p className="text-2xl font-bold text-cyan-900">
                {((stats.avg_bonus_hit_rate || 0) * 100).toFixed(2)}%
              </p>
            </div>
          )}
          <div className="bg-green-50 p-4 rounded">
            <p className="text-sm text-green-700">Lift vs Aléatoire</p>
            <p className={`text-2xl font-bold ${(stats.lift_vs_random || 0) > 1 ? 'text-green-900' : 'text-red-900'}`}>
              {(stats.lift_vs_random || 0).toFixed(2)}x
            </p>
          </div>
          <div className="bg-purple-50 p-4 rounded">
            <p className="text-sm text-purple-700">Total Hits</p>
            <p className="text-2xl font-bold text-purple-900">{stats.total_main_hits || stats.total_hits || 0}/{stats.total_bonus_hits || 0}</p>
            <p className="text-xs text-purple-600">Main/Bonus</p>
          </div>
        </div>

        {/* Predictions Timeline */}
        <div className="mb-6">
          <h4 className="font-medium text-gray-900 mb-3">📈 Évolution du taux de réussite</h4>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={predictions.map((p: any, idx: number) => ({
              index: idx + 1,
              mainHitRate: (p.main_hit_rate || p.hit_rate || 0) * 100,
              bonusHitRate: (p.bonus_hit_rate || 0) * 100,
              date: new Date(p.draw_date).toLocaleDateString()
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" label={{ value: 'Tirage #', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Taux de réussite (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="mainHitRate" stroke="#3b82f6" strokeWidth={2} name="Principaux" />
              {hasBonus && <Line type="monotone" dataKey="bonusHitRate" stroke="#06b6d4" strokeWidth={2} name="Bonus" />}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed Predictions Table */}
        <div>
          <h4 className="font-medium text-gray-900 mb-3">
            📋 Détail des prédictions (tous les tirages)
          </h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Prédits (Principaux)</th>
                  {hasBonus && <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Prédits (Bonus)</th>}
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Réels (Principaux)</th>
                  {hasBonus && <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Réels (Bonus)</th>}
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Hits</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Taux</th>
                  {(modelName === 'ANTI' || modelName === 'ANTI2') && <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Divisions</th>}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {predictions.slice().reverse().map((pred: any, idx: number) => {
                  if (!pred) return null;
                  const mainNumbers = pred.predicted_main || pred.predicted || [];
                  const bonusNumbers = pred.predicted_bonus || [];
                  const actualMain = pred.actual_main || pred.actual || [];
                  const actualBonus = pred.actual_bonus || [];
                  const mainHits = pred.main_hits || pred.hits || [];
                  const bonusHits = pred.bonus_hits || [];
                  
                  return (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-700">
                      {pred.draw_date ? pred.draw_date.split('T')[0] : '-'}
                    </td>
                    
                    {/* Prédits Principaux */}
                    <td className="px-4 py-2">
                      {(modelName === 'ANTI' || modelName === 'ANTI2') && pred.all_combinations && pred.all_combinations.length > 0 ? (
                        <div className="flex flex-col gap-1">
                          <div className="text-xs font-semibold text-gray-700">{pred.all_combinations.length} combos:</div>
                          <div className="space-y-1">
                            {pred.all_combinations.map((combo: any, comboIdx: number) => (
                              <div key={comboIdx} className="flex items-center gap-1">
                                <span className="text-xs text-gray-500 font-mono">#{comboIdx + 1}</span>
                                <div className="flex flex-wrap gap-1">
                                  {(combo.main_combination || combo.combination || []).map((num: number) => {
                                    const isMainHit = (combo.main_hits || combo.hits || []).includes(num);
                                    const isBonusHit = (pred.actual_bonus || []).includes(num);
                                    return (
                                      <span
                                        key={num}
                                        className={`inline-flex items-center justify-center w-6 h-6 text-xs font-medium ${
                                          isMainHit
                                            ? 'bg-green-100 text-green-800 border border-green-500 rounded'
                                            : isBonusHit
                                              ? 'bg-red-50 text-red-700 border-2 border-red-500 rounded-full'
                                              : 'bg-white text-gray-600 border border-gray-300 rounded'
                                        }`}
                                      >
                                        {num}
                                      </span>
                                    );
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-wrap gap-1">
                          {(pred.predicted_main || pred.predicted || []).map((num: number) => {
                            const isMainHit = (pred.main_hits || pred.hits || []).includes(num);
                            const isBonusHit = (pred.actual_bonus || []).includes(num);
                            return (
                              <span
                                key={num}
                                className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-medium ${
                                  isMainHit
                                    ? 'bg-green-100 text-green-800 border-2 border-green-500'
                                    : isBonusHit
                                      ? 'bg-red-50 text-red-700 border-2 border-red-500 rounded-full'
                                      : 'bg-gray-100 text-gray-600'
                                }`}
                              >
                                {num}
                              </span>
                            );
                          })}
                        </div>
                      )}
                    </td>
                    
                    {/* Prédits Bonus */}
                    {hasBonus && (
                      <td className="px-4 py-2">
                        {(modelName === 'ANTI' || modelName === 'ANTI2') && pred.all_combinations && pred.all_combinations.length > 0 ? (
                          <div className="space-y-1">
                            {pred.all_combinations.map((combo: any, comboIdx: number) => (
                              <div key={comboIdx} className="flex flex-wrap gap-1">
                                {(combo.bonus_combination || []).map((num: number) => (
                                  <span
                                    key={num}
                                    className={`inline-flex items-center justify-center w-6 h-6 rounded text-xs font-medium ${
                                      (combo.bonus_hits || []).includes(num)
                                        ? 'bg-cyan-100 text-cyan-800 border border-cyan-500'
                                        : 'bg-white text-gray-600 border border-gray-300'
                                    }`}
                                  >
                                    {num}
                                  </span>
                                ))}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="flex flex-wrap gap-1">
                            {(pred.predicted_bonus || []).map((num: number) => (
                              <span
                                key={num}
                                className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-medium ${
                                  (pred.bonus_hits || []).includes(num)
                                    ? 'bg-cyan-100 text-cyan-800 border-2 border-cyan-500'
                                    : 'bg-gray-100 text-gray-600'
                                }`}
                              >
                                {num}
                              </span>
                            ))}
                          </div>
                        )}
                      </td>
                    )}
                    
                    {/* Réels Principaux */}
                    <td className="px-4 py-2">
                      <div className="flex flex-wrap gap-1">
                        {(pred.actual_main || pred.actual || []).map((num: number) => (
                          <span
                            key={num}
                            className="inline-flex items-center justify-center w-7 h-7 rounded bg-blue-100 text-blue-800 text-xs font-medium"
                          >
                            {num}
                          </span>
                        ))}
                      </div>
                    </td>
                    
                    {/* Réels Bonus */}
                    {hasBonus && (
                      <td className="px-4 py-2">
                        <div className="flex flex-wrap gap-1">
                          {(pred.actual_bonus || []).map((num: number) => (
                            <span
                              key={num}
                              className="inline-flex items-center justify-center w-7 h-7 rounded bg-cyan-100 text-cyan-800 text-xs font-medium"
                            >
                              {num}
                            </span>
                          ))}
                        </div>
                      </td>
                    )}
                    
                    {/* Hits */}
                    <td className="px-4 py-2">
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-1">
                          {(pred.n_main_hits || pred.n_hits || 0) > 0 ? (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-600" />
                          )}
                          <span className="text-xs font-medium">{pred.n_main_hits || pred.n_hits || 0}/{pred.n_main_drawn || pred.n_drawn || 0}</span>
                        </div>
                        {pred.n_bonus_drawn > 0 && (
                          <div className="flex items-center gap-1">
                            {pred.n_bonus_hits > 0 ? (
                              <CheckCircle className="w-4 h-4 text-cyan-600" />
                            ) : (
                              <XCircle className="w-4 h-4 text-gray-400" />
                            )}
                            <span className="text-xs font-medium text-cyan-700">{pred.n_bonus_hits}/{pred.n_bonus_drawn}</span>
                          </div>
                        )}
                      </div>
                    </td>
                    
                    {/* Taux */}
                    <td className="px-4 py-2 whitespace-nowrap">
                      <div className="flex flex-col gap-1">
                        <span className={`text-xs font-medium ${(pred.main_hit_rate || pred.hit_rate || 0) > 0.2 ? 'text-green-600' : 'text-gray-600'}`}>
                          {((pred.main_hit_rate || pred.hit_rate || 0) * 100).toFixed(1)}%
                        </span>
                        {pred.n_bonus_drawn > 0 && (
                          <span className={`text-xs font-medium ${pred.bonus_hit_rate > 0.2 ? 'text-cyan-600' : 'text-gray-600'}`}>
                            {(pred.bonus_hit_rate * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </td>
                    
                    {/* Divisions Summary (ANTI/ANTI2 only) */}
                    {(modelName === 'ANTI' || modelName === 'ANTI2') && (
                      <td className="px-4 py-2">
                        {pred.division_summary && Object.keys(pred.division_summary).length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(pred.division_summary)
                              .sort(([a], [b]) => parseInt(a.replace('div_', '')) - parseInt(b.replace('div_', '')))
                              .map(([divKey, count]: [string, any]) => (
                                <span 
                                  key={divKey} 
                                  className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                    divKey === 'div_1' ? 'bg-yellow-100 text-yellow-800' :
                                    divKey === 'div_2' ? 'bg-orange-100 text-orange-800' :
                                    divKey === 'div_3' ? 'bg-blue-100 text-blue-800' :
                                    divKey === 'div_4' ? 'bg-green-100 text-green-800' :
                                    divKey === 'div_5' ? 'bg-purple-100 text-purple-800' :
                                    'bg-gray-100 text-gray-800'
                                  }`}
                                >
                                  D{divKey.replace('div_', '')}: {count}
                                </span>
                              ))}
                          </div>
                        ) : (
                          <span className="text-xs text-gray-400">-</span>
                        )}
                      </td>
                    )}
                  </tr>
                );})}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">🎯 Validation par Backtest</h1>
        <p className="text-gray-600">
          Évaluation walk-forward : confrontation des Top N prédictions avec les tirages réels
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Configuration du Backtest</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Jeu</label>
            <select
              value={selectedGameId}
              onChange={(e) => setSelectedGameId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {games.map((game) => (
                <option key={game.id} value={game.id}>
                  {game.name}
                </option>
              ))}
            </select>
            {/* Game configuration display */}
            {selectedGameId && games.find(g => g.id === selectedGameId) && (() => {
              const game = games.find(g => g.id === selectedGameId);
              const rules = game?.rules_json;
              return rules ? (
                <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-600">
                  <div><strong>Main:</strong> Pick {rules.main?.pick || '?'}, Draw {rules.main?.drawn || rules.main?.pick || '?'} from {rules.main?.min || 1}-{rules.main?.max || 49}</div>
                  {rules.bonus?.enabled && (
                    <div><strong>Bonus:</strong> Pick {rules.bonus.pick || 0}, Draw {rules.bonus.drawn || rules.bonus.pick || 0} {rules.bonus.separate_pool ? `from ${rules.bonus.min}-${rules.bonus.max}` : '(same pool)'}</div>
                  )}
                </div>
              ) : null;
            })()}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Nombre de tirages à tester
            </label>
            <input
              type="number"
              value={nTestDraws}
              onChange={(e) => setNTestDraws(parseInt(e.target.value))}
              min="5"
              max="100"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Top N (nombre de prédictions)
            </label>
            <input
              type="number"
              value={topN}
              onChange={(e) => setTopN(parseInt(e.target.value))}
              min="5"
              max="20"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Combinaisons Anti-Consensus
            </label>
            <input
              type="number"
              value={nCombinations}
              onChange={(e) => setNCombinations(parseInt(e.target.value))}
              min="5"
              max="50"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
            <p className="text-xs text-gray-500 mt-1">Nombre de combinaisons à générer pour le modèle ANTI</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max numéros principaux en commun (ANTI2)
            </label>
            <input
              type="number"
              value={maxCommonMain}
              onChange={(e) => setMaxCommonMain(parseInt(e.target.value))}
              min="0"
              max="5"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
            <p className="text-xs text-gray-500 mt-1">Nombre max de numéros principaux identiques entre combinaisons</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max numéros bonus en commun (ANTI2)
            </label>
            <input
              type="number"
              value={maxCommonBonus}
              onChange={(e) => setMaxCommonBonus(parseInt(e.target.value))}
              min="0"
              max="3"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
            <p className="text-xs text-gray-500 mt-1">Nombre max de numéros bonus identiques entre combinaisons</p>
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Modèles à tester</label>
          <div className="flex flex-wrap gap-2">
            {['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M20', 'ANTI', 'ANTI2'].map((model) => (
              <button
                key={model}
                onClick={() => toggleModel(model)}
                className={`px-4 py-2 rounded-md font-medium transition-colors ${
                  selectedModels.includes(model)
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {model}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleRunBacktest}
          disabled={loading || !selectedGameId || selectedModels.length === 0}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
        >
          {loading ? 'Exécution en cours...' : '🚀 Lancer le Backtest'}
        </button>
      </div>

      {/* Next Draw Prediction Section */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg shadow p-6 mb-6 border border-purple-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-purple-900 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-purple-600" />
            Prédiction Anti-Consensus v2 (ANTI2) pour le prochain tirage
          </h2>
          <button
            onClick={loadNextPrediction}
            disabled={loadingPrediction || !selectedGameId}
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium flex items-center gap-2"
          >
            {loadingPrediction ? 'Chargement...' : '✨ Générer Prédiction'}
          </button>
        </div>
        
        {nextPrediction && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg p-4 border border-purple-100">
              <h3 className="font-medium text-gray-900 mb-3">🎯 Numéros prédits</h3>
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="text-sm text-gray-600 mr-2">Principaux:</span>
                {nextPrediction.main_numbers.map((num, idx) => (
                  <span key={idx} className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-purple-600 text-white font-bold text-lg">
                    {num}
                  </span>
                ))}
              </div>
              {nextPrediction.bonus_numbers.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  <span className="text-sm text-gray-600 mr-2">Bonus:</span>
                  {nextPrediction.bonus_numbers.map((num, idx) => (
                    <span key={idx} className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-amber-500 text-white font-bold text-lg">
                      {num}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg p-4 border border-purple-100">
                <h4 className="font-medium text-gray-900 mb-2">📊 Informations</h4>
                <p className="text-sm text-gray-600">Tirages analysés: <span className="font-medium">{nextPrediction.total_draws_analyzed}</span></p>
                <p className="text-sm text-gray-600">Modèles utilisés: <span className="font-medium">{nextPrediction.models_used.join(', ')}</span></p>
              </div>
              
              <div className="bg-white rounded-lg p-4 border border-purple-100">
                <h4 className="font-medium text-gray-900 mb-2">🚫 Numéros exclus (prédits par consensus)</h4>
                <div className="flex flex-wrap gap-1">
                  {nextPrediction.excluded_main.slice(0, 15).map((num, idx) => (
                    <span key={idx} className="inline-flex items-center justify-center w-7 h-7 rounded bg-gray-200 text-gray-600 text-sm font-medium">
                      {num}
                    </span>
                  ))}
                  {nextPrediction.excluded_main.length > 15 && (
                    <span className="text-sm text-gray-500">+{nextPrediction.excluded_main.length - 15} autres</span>
                  )}
                </div>
              </div>
            </div>

            {nextPrediction.combinations.length > 1 && (
              <div className="bg-white rounded-lg p-4 border border-purple-100">
                <h4 className="font-medium text-gray-900 mb-3">🎲 Toutes les combinaisons générées ({nextPrediction.combinations.length})</h4>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {nextPrediction.combinations.map((combo: any, idx: number) => (
                    <div key={idx} className="flex items-center gap-1">
                      <span className="text-gray-500 text-sm w-8">#{idx + 1}</span>
                      {Array.isArray(combo.main_numbers) && combo.main_numbers.map((num: number, numIdx: number) => (
                        <span 
                          key={numIdx} 
                          className="inline-flex items-center justify-center w-8 h-8 border-2 border-green-500 rounded text-sm font-medium text-gray-900"
                        >
                          {num}
                        </span>
                      ))}
                      {combo.bonus_numbers && combo.bonus_numbers.length > 0 && combo.bonus_numbers.map((num: number, numIdx: number) => (
                        <span 
                          key={`bonus-${numIdx}`} 
                          className="inline-flex items-center justify-center w-8 h-8 border-2 border-amber-500 rounded text-sm font-medium text-amber-700 ml-2"
                        >
                          {num}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {!nextPrediction && !loadingPrediction && (
          <p className="text-gray-500 text-center py-4">
            Cliquez sur "Générer Prédiction" pour obtenir une prédiction Anti-Consensus basée sur tout l'historique des tirages.
          </p>
        )}
      </div>

      {result && result.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-800">{result.error}</p>
        </div>
      )}

      {result && result.results?.backtest && (
        <div className="space-y-6">
          {/* Comparison Chart */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <Award className="w-5 h-5 mr-2" />
              Comparaison des Modèles
            </h2>
            
            <div className="mb-6">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getComparisonChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="avgHitRate" fill="#3b82f6" name="Taux de réussite moyen (%)" />
                  <Bar dataKey="maxHitRate" fill="#f59e0b" name="Taux de réussite max (%)" />
                  <Bar dataKey="liftVsRandom" fill="#10b981" name="Lift vs Aléatoire" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Ranking Table */}
            {result.results.backtest.comparison?.ranking && (
              <div className="overflow-x-auto">
                <h3 className="font-medium text-gray-900 mb-3">🏆 Classement des Modèles</h3>
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Rang</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Modèle</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Taux moyen</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Lift</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Total Hits</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {result.results.backtest.comparison.ranking.map((rank: any, idx: number) => (
                      <tr key={idx} className={idx === 0 ? 'bg-yellow-50' : ''}>
                        <td className="px-4 py-2 whitespace-nowrap">
                          {idx === 0 && <span className="text-2xl">🥇</span>}
                          {idx === 1 && <span className="text-2xl">🥈</span>}
                          {idx === 2 && <span className="text-2xl">🥉</span>}
                          {idx > 2 && <span className="font-medium text-gray-600">#{idx + 1}</span>}
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap font-medium">{rank.model}</td>
                        <td className="px-4 py-2 whitespace-nowrap">
                          <span className="text-blue-600 font-medium">
                            {(rank.avg_hit_rate * 100).toFixed(2)}%
                          </span>
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap">
                          <span className={rank.lift_vs_random > 1 ? 'text-green-600 font-medium' : 'text-gray-600'}>
                            {rank.lift_vs_random.toFixed(2)}x
                          </span>
                        </td>
                        <td className="px-4 py-2 whitespace-nowrap">{rank.total_hits}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Individual Model Results */}
          {Object.entries(result.results.backtest.models).map(([modelName, modelData]: [string, any]) =>
            renderModelResults(modelName, modelData)
          )}
        </div>
      )}
    </div>
  )
}
