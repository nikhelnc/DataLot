import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { BarChart, Bar, LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'
import { TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react'
import { gamesApi, analysesApi } from '../api/client'

export default function AdvancedModels() {
  const { t } = useTranslation()
  const [games, setGames] = useState<any[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [selectedModels, setSelectedModels] = useState<string[]>(['M5', 'M6', 'M9', 'M10'])

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

  const runAdvancedModels = async () => {
    if (!selectedGameId) return

    setLoading(true)
    try {
      const response = await analysesApi.run({
        game_id: selectedGameId,
        analysis_name: 'advanced_models_v1',
        params: {
          models: selectedModels
        }
      })
      setResult(response.data.results)
    } catch (error) {
      console.error('Failed to run advanced models:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleModel = (model: string) => {
    if (selectedModels.includes(model)) {
      setSelectedModels(selectedModels.filter(m => m !== model))
    } else {
      setSelectedModels([...selectedModels, model])
    }
  }

  const renderM5Results = (m5Data: any) => {
    if (!m5Data || m5Data.error) return null

    const cooc = m5Data.cooccurrence
    const topPairs = cooc.top_pairs?.slice(0, 10) || []

    // Extract top numbers from pairs
    const numberFrequency: { [key: number]: number } = {}
    topPairs.forEach((pair: any) => {
      numberFrequency[pair.num1] = (numberFrequency[pair.num1] || 0) + pair.delta
      numberFrequency[pair.num2] = (numberFrequency[pair.num2] || 0) + pair.delta
    })
    const topNumbers = Object.entries(numberFrequency)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([num, score]) => ({ number: parseInt(num), score }))

    return (
      <div className="space-y-6">
        <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4">
          <h4 className="font-semibold text-blue-900 mb-3">🎯 Top 10 Numéros Analytiques (Co-occurrence)</h4>
          <div className="flex flex-wrap gap-2">
            {topNumbers.map((item, idx) => (
              <div key={idx} className="flex flex-col items-center">
                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-600 text-white text-lg font-bold shadow-lg">
                  {item.number}
                </span>
                <span className="text-xs text-blue-700 mt-1">Score: {item.score.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">📊 Top 10 paires sur-représentées</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Paire</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Observé</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Attendu</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Delta</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">p-value (FDR)</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topPairs.map((pair: any, idx: number) => (
                  <tr key={idx}>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className="inline-flex items-center space-x-1">
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">{pair.num1}</span>
                        <span>-</span>
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">{pair.num2}</span>
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">{pair.observed.toFixed(1)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">{pair.expected.toFixed(1)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className={pair.delta > 0 ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                        {pair.delta > 0 ? '+' : ''}{pair.delta.toFixed(1)}
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className={pair.p_value_fdr < 0.05 ? 'text-red-600 font-medium' : 'text-gray-600'}>
                        {pair.p_value_fdr?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">📉 Graphique des deltas</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topPairs}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={(d: any) => `${d.num1}-${d.num2}`} angle={-45} textAnchor="end" height={80} />
              <YAxis label={{ value: 'Delta vs attendu', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="delta" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    )
  }

  const renderM6Results = (m6Data: any) => {
    if (!m6Data || m6Data.error) return null

    const gapsStreaks = m6Data.gaps_streaks
    const topAtypical = gapsStreaks.top_atypical?.slice(0, 10) || []
    const gapHist = gapsStreaks.gap_histogram

    // Extract top numbers with largest positive gaps (numbers "overdue")
    const overdueNumbers = topAtypical
      .filter((s: any) => s.delta_gap && s.delta_gap > 0)
      .sort((a: any, b: any) => b.delta_gap - a.delta_gap)
      .slice(0, 10)

    return (
      <div className="space-y-6">
        <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
          <h4 className="font-semibold text-green-900 mb-3">🎯 Top 10 Numéros Analytiques (Gaps - "En retard")</h4>
          <div className="flex flex-wrap gap-2">
            {overdueNumbers.map((stat: any, idx: number) => (
              <div key={idx} className="flex flex-col items-center">
                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-600 text-white text-lg font-bold shadow-lg">
                  {stat.number}
                </span>
                <span className="text-xs text-green-700 mt-1">Gap: +{stat.delta_gap.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">📊 Top 10 numéros avec gaps atypiques</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Numéro</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Gap moyen</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Gap attendu</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Delta</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Gap max</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">KS p-value</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topAtypical.map((stat: any, idx: number) => (
                  <tr key={idx}>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded font-medium">{stat.number}</span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">{stat.mean_gap?.toFixed(2) || 'N/A'}</td>
                    <td className="px-4 py-2 whitespace-nowrap">{stat.expected_gap.toFixed(2)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className={stat.delta_gap > 0 ? 'text-red-600 font-medium' : 'text-green-600 font-medium'}>
                        {stat.delta_gap > 0 ? '+' : ''}{stat.delta_gap?.toFixed(2) || 'N/A'}
                      </span>
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">{stat.max_gap || 'N/A'}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <span className={stat.ks_pval && stat.ks_pval < 0.05 ? 'text-red-600 font-medium' : 'text-gray-600'}>
                        {stat.ks_pval?.toFixed(4) || 'N/A'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {gapHist && gapHist.counts && gapHist.counts.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-900 mb-2">📊 Distribution globale des gaps</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={gapHist.counts.map((count: number, idx: number) => ({
                bin: idx,
                count: count
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" label={{ value: 'Taille du gap', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Fréquence', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    )
  }

  const renderM9Results = (m9Data: any) => {
    if (!m9Data || m9Data.error) return null

    const metatest = m9Data.metatest
    const qqPlot = metatest.qq_plot

    return (
      <div className="space-y-6">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">🔍 Verdict</h4>
          <p className="text-blue-800">{metatest.interpretation}</p>
          <div className="mt-2 grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-blue-700">KS statistic:</span>
              <span className="ml-2 font-medium">{metatest.ks_statistic.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-sm text-blue-700">KS p-value:</span>
              <span className="ml-2 font-medium">{metatest.ks_pvalue.toFixed(4)}</span>
            </div>
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">📈 QQ Plot (p-values vs Uniforme[0,1])</h4>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number" 
                domain={[0, 1]} 
                label={{ value: 'Quantiles théoriques (Uniforme)', position: 'insideBottom', offset: -5 }} 
              />
              <YAxis 
                dataKey="y" 
                type="number" 
                domain={[0, 1]} 
                label={{ value: 'Quantiles observés (p-values)', angle: -90, position: 'insideLeft' }} 
              />
              <Tooltip />
              <Scatter 
                name="P-values" 
                data={qqPlot.theoretical.map((t: number, idx: number) => ({
                  x: t,
                  y: qqPlot.observed[idx]
                }))} 
                fill="#3b82f6" 
              />
              <Line 
                type="linear" 
                dataKey="y" 
                data={[{x: 0, y: 0}, {x: 1, y: 1}]} 
                stroke="#ef4444" 
                strokeWidth={2} 
                dot={false} 
                name="Ligne y=x (parfait)"
              />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-sm text-gray-600 mt-2">
            ℹ️ Si les points suivent la ligne rouge, les p-values sont uniformes (attendu sous H0)
          </p>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">📊 Comptage des résultats significatifs</h4>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(metatest.significance_counts).map(([threshold, counts]: [string, any]) => (
              <div key={threshold} className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">{threshold.replace('_', ' < ')}</div>
                <div className="text-2xl font-bold text-gray-900">{counts.observed}</div>
                <div className="text-sm text-gray-500">
                  Attendu: {counts.expected.toFixed(1)} 
                  <span className={counts.delta > 0 ? 'text-red-600 ml-2' : 'text-green-600 ml-2'}>
                    ({counts.delta > 0 ? '+' : ''}{counts.delta.toFixed(1)})
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderM10Results = (m10Data: any) => {
    if (!m10Data || m10Data.error) return null

    const ensemble = m10Data.ensemble
    const weights = ensemble.weights
    const numberProbs = m10Data.number_probs || {}

    // Extract top 10 numbers by combined probability
    const topNumbers = Object.entries(numberProbs)
      .map(([num, prob]) => ({ number: parseInt(num), probability: prob as number }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 10)

    return (
      <div className="space-y-6">
        <div className="bg-purple-50 border-2 border-purple-300 rounded-lg p-4">
          <h4 className="font-semibold text-purple-900 mb-3">🎯 Top 10 Numéros Analytiques (Ensemble - Probabilités combinées)</h4>
          <div className="flex flex-wrap gap-2">
            {topNumbers.map((item, idx) => (
              <div key={idx} className="flex flex-col items-center">
                <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-600 text-white text-lg font-bold shadow-lg">
                  {item.number}
                </span>
                <span className="text-xs text-purple-700 mt-1">P: {(item.probability * 100).toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="text-sm text-blue-700 mb-1">Brier Score</div>
            <div className="text-2xl font-bold text-blue-900">{ensemble.brier_score.toFixed(6)}</div>
            <div className="text-sm text-blue-600">
              Baseline: {ensemble.baseline_brier.toFixed(6)}
            </div>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <div className="text-sm text-green-700 mb-1">Delta vs Baseline</div>
            <div className={`text-2xl font-bold ${ensemble.delta_brier < 0 ? 'text-green-900' : 'text-red-900'}`}>
              {ensemble.delta_brier > 0 ? '+' : ''}{ensemble.delta_brier.toFixed(6)}
            </div>
            <div className="text-sm text-green-600">
              {ensemble.delta_brier < 0 ? '✓ Amélioration' : '✗ Dégradation'}
            </div>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <div className="text-sm text-purple-700 mb-1">Lift</div>
            <div className="text-2xl font-bold text-purple-900">{ensemble.lift.toFixed(3)}x</div>
            <div className="text-sm text-purple-600">
              ECE: {ensemble.calibration.ece?.toFixed(4) || 'N/A'}
            </div>
          </div>
        </div>

        <div>
          <h4 className="font-medium text-gray-900 mb-2">⚖️ Poids des modèles</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={Object.entries(weights).map(([model, weight]) => ({
              model,
              weight: (weight as number) * 100
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis label={{ value: 'Poids (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="weight" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {ensemble.calibration && ensemble.calibration.pred_probs && ensemble.calibration.pred_probs.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-900 mb-2">📊 Courbe de calibration</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ensemble.calibration.pred_probs.map((pred: number, idx: number) => ({
                pred,
                obs: ensemble.calibration.obs_freqs[idx]
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="pred" 
                  label={{ value: 'Probabilité prédite', position: 'insideBottom', offset: -5 }} 
                />
                <YAxis 
                  label={{ value: 'Fréquence observée', angle: -90, position: 'insideLeft' }} 
                />
                <Tooltip />
                <Line type="monotone" dataKey="obs" stroke="#3b82f6" strokeWidth={2} name="Observé" />
                <Line type="linear" dataKey="pred" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Parfait" />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-sm text-gray-600 mt-2">
              ℹ️ Si la courbe bleue suit la ligne rouge, les probabilités sont bien calibrées
            </p>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="p-6 max-w-[88rem] mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Modèles Avancés (M5-M10)</h1>
        <p className="text-gray-600">
          Analyses statistiques avancées : co-occurrences, gaps & streaks, meta-tests, ensemble stacking
        </p>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
        <div className="flex items-start">
          <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 mr-3" />
          <div>
            <h3 className="font-medium text-yellow-900 mb-1">⚠️ Important : Interprétation Scientifique</h3>
            <ul className="text-sm text-yellow-800 space-y-1">
              <li>• Ces analyses sont à des fins de recherche et diagnostic uniquement</li>
              <li>• Les résultats ne garantissent aucun avantage dans les tirages futurs</li>
              <li>• La signification statistique est testée - les résultats non significatifs sont signalés</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Configuration</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Jeu</label>
            <select
              value={selectedGameId}
              onChange={(e) => setSelectedGameId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {games.map((game) => (
                <option key={game.id} value={game.id}>{game.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Modèles à exécuter</label>
            <div className="grid grid-cols-2 gap-2">
              {['M5', 'M6', 'M9', 'M10'].map((model) => (
                <label key={model} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model)}
                    onChange={() => toggleModel(model)}
                    className="rounded border-gray-300"
                  />
                  <span className="text-sm text-gray-700">
                    {model === 'M5' && 'M5 - Co-occurrence'}
                    {model === 'M6' && 'M6 - Gaps & Streaks'}
                    {model === 'M9' && 'M9 - Meta-test p-values'}
                    {model === 'M10' && 'M10 - Ensemble'}
                  </span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={runAdvancedModels}
            disabled={loading || !selectedGameId || selectedModels.length === 0}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Calcul en cours...' : 'Lancer les analyses'}
          </button>
        </div>
      </div>

      {result && result.advanced_models && (
        <div className="space-y-6">
          {result.advanced_models.M5 && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">M5 - Co-occurrence Matrix</h2>
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <p className="text-sm text-gray-600 mb-4">{result.advanced_models.M5.explain}</p>
              {result.advanced_models.M5.warnings && result.advanced_models.M5.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4">
                  {result.advanced_models.M5.warnings.map((w: string, idx: number) => (
                    <p key={idx} className="text-sm text-yellow-800">⚠️ {w}</p>
                  ))}
                </div>
              )}
              {renderM5Results(result.advanced_models.M5)}
            </div>
          )}

          {result.advanced_models.M6 && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">M6 - Gaps & Streaks</h2>
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <p className="text-sm text-gray-600 mb-4">{result.advanced_models.M6.explain}</p>
              {result.advanced_models.M6.warnings && result.advanced_models.M6.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4">
                  {result.advanced_models.M6.warnings.map((w: string, idx: number) => (
                    <p key={idx} className="text-sm text-yellow-800">⚠️ {w}</p>
                  ))}
                </div>
              )}
              {renderM6Results(result.advanced_models.M6)}
            </div>
          )}

          {result.advanced_models.M9 && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">M9 - Meta-test des p-values</h2>
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <p className="text-sm text-gray-600 mb-4">{result.advanced_models.M9.explain}</p>
              {result.advanced_models.M9.warnings && result.advanced_models.M9.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4">
                  {result.advanced_models.M9.warnings.map((w: string, idx: number) => (
                    <p key={idx} className="text-sm text-yellow-800">⚠️ {w}</p>
                  ))}
                </div>
              )}
              {renderM9Results(result.advanced_models.M9)}
            </div>
          )}

          {result.advanced_models.M10 && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900">M10 - Ensemble Stacking</h2>
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <p className="text-sm text-gray-600 mb-4">{result.advanced_models.M10.explain}</p>
              {result.advanced_models.M10.warnings && result.advanced_models.M10.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mb-4">
                  {result.advanced_models.M10.warnings.map((w: string, idx: number) => (
                    <p key={idx} className="text-sm text-yellow-800">⚠️ {w}</p>
                  ))}
                </div>
              )}
              {renderM10Results(result.advanced_models.M10)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
