import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { gamesApi, analysesApi, drawsApi } from '../api/client'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { Calculator, TrendingUp, Award, Info, Download } from 'lucide-react'
import { exportToJSON } from '../utils/exportData'

export default function Probabilities() {
  const { t } = useTranslation()
  const [games, setGames] = useState<any[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [selectedNumbers, setSelectedNumbers] = useState<number[]>([])
  const [combinationProb, setCombinationProb] = useState<number | null>(null)
  const [draws, setDraws] = useState<any[]>([])
  const [showCalculator, setShowCalculator] = useState(false)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGameId) {
      loadDraws()
    }
  }, [selectedGameId])

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

  const loadDraws = async () => {
    if (!selectedGameId) return
    try {
      const response = await drawsApi.list(selectedGameId, { limit: 1000 })
      setDraws(response.data)
    } catch (error) {
      console.error('Failed to load draws:', error)
    }
  }

  const handleRunForecast = async () => {
    if (!selectedGameId) return

    setLoading(true)
    try {
      const response = await analysesApi.run({
        game_id: selectedGameId,
        analysis_name: 'forecast_probabilities_v1',
        params: { models: ['M0', 'M1', 'M2'] },
      })
      setResult(response.data)
    } catch (error: any) {
      console.error('Forecast failed:', error)
      setResult({ error: error.response?.data?.detail || 'Forecast failed' })
    } finally {
      setLoading(false)
    }
  }

  const getProbabilityChartData = (modelData: any) => {
    if (!modelData?.number_probs) return []
    
    return Object.entries(modelData.number_probs)
      .map(([num, prob]) => ({
        number: num,
        probability: (prob as number) * 100,
        baseline: modelData.baseline_probs?.[num] ? (modelData.baseline_probs[num] as number) * 100 : 0,
      }))
      .sort((a, b) => parseInt(a.number) - parseInt(b.number))
  }

  const getTopNNumbers = (modelData: any, n: number = 10) => {
    if (!modelData?.number_probs) return []
    
    return Object.entries(modelData.number_probs)
      .map(([num, prob]) => ({
        number: parseInt(num),
        probability: prob as number,
        baseline: modelData.baseline_probs?.[num] || 0,
        delta: (prob as number) - (modelData.baseline_probs?.[num] || 0),
        credible_lower: modelData.credible_interval_95?.[num]?.[0],
        credible_upper: modelData.credible_interval_95?.[num]?.[1],
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, n)
  }

  const getModelComparisonData = () => {
    if (!result?.results?.probabilities) return []
    
    const models = Object.entries(result.results.probabilities)
    return models.map(([name, data]: [string, any]) => ({
      model: name,
      brierScore: data.evaluation?.brier_score || 0,
      lift: data.evaluation?.lift || 0,
      ece: data.evaluation?.ece || 0
    }))
  }

  const calculateCombinationProbability = () => {
    if (selectedNumbers.length === 0 || !result?.results?.probabilities) return
    
    const modelData = Object.values(result.results.probabilities)[0] as any
    if (!modelData?.number_probs) return
    
    let prob = 1
    selectedNumbers.forEach(num => {
      const numProb = modelData.number_probs[num] || 0
      prob *= numProb
    })
    
    setCombinationProb(prob)
  }

  const toggleNumber = (num: number) => {
    if (selectedNumbers.includes(num)) {
      setSelectedNumbers(selectedNumbers.filter(n => n !== num))
    } else {
      const game = games.find(g => g.id === selectedGameId)
      const maxNumbers = game?.rules_json?.numbers?.count || 7
      if (selectedNumbers.length < maxNumbers) {
        setSelectedNumbers([...selectedNumbers, num].sort((a, b) => a - b))
      }
    }
  }

  const handleExportProbabilities = () => {
    if (!result) return
    exportToJSON(result.results, `probabilities-forecast-${new Date().toISOString().split('T')[0]}.json`)
  }

  const getHistoricalFrequency = () => {
    if (draws.length === 0) return {}
    
    const freq: Record<number, number> = {}
    draws.forEach(draw => {
      draw.numbers.forEach((num: number) => {
        freq[num] = (freq[num] || 0) + 1
      })
    })
    
    const total = draws.length
    const normalized: Record<number, number> = {}
    Object.entries(freq).forEach(([num, count]) => {
      normalized[parseInt(num)] = count / total
    })
    
    return normalized
  }

  return (
    <div className="px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">{t('probabilities.title')}</h1>
        {result && (
          <button
            onClick={handleExportProbabilities}
            className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <Download className="w-4 h-4 mr-2" />
            {t('probabilities.exportResults')}
          </button>
        )}
      </div>

      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <h3 className="text-sm font-semibold text-red-900 mb-2">
          {t('probabilities.warning.title')}
        </h3>
        <ul className="text-xs text-red-800 space-y-1">
          <li>{t('probabilities.warning.line1')}</li>
          <li>{t('probabilities.warning.line2')}</li>
          <li>{t('probabilities.warning.line3')}</li>
          <li>{t('probabilities.warning.line4')}</li>
        </ul>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('probabilities.selectGame')}
            </label>
            <select
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={selectedGameId}
              onChange={(e) => setSelectedGameId(e.target.value)}
            >
              {games.map(game => (
                <option key={game.id} value={game.id}>
                  {game.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleRunForecast}
            disabled={!selectedGameId || loading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
          >
            {loading ? t('probabilities.computing') : t('probabilities.runForecast')}
          </button>
        </div>
      </div>

      {result && result.results?.probabilities && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2" />
              {t('probabilities.modelComparison')}
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {getModelComparisonData().map((model, idx) => (
                <div key={idx} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
                  <h3 className="font-semibold text-blue-900 mb-3">{model.model}</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-blue-700">{t('probabilities.brierScore')}:</span>
                      <span className="font-medium text-blue-900">{model.brierScore.toFixed(6)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-blue-700">{t('probabilities.lift')}:</span>
                      <span className={`font-medium ${model.lift > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {model.lift.toFixed(6)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-blue-700">{t('probabilities.ece')}:</span>
                      <span className="font-medium text-blue-900">{model.ece.toFixed(6)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={getModelComparisonData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="lift" fill="#10b981" name="Lift" />
                <Bar dataKey="ece" fill="#f59e0b" name="ECE" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Calculator className="w-5 h-5 mr-2" />
                {t('probabilities.calculator')}
              </h2>
              <button
                onClick={() => setShowCalculator(!showCalculator)}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                {showCalculator ? t('probabilities.hide') : t('probabilities.show')}
              </button>
            </div>
            
            {showCalculator && (
              <div>
                <p className="text-sm text-gray-600 mb-4">
                  {t('probabilities.selectNumbers')}
                </p>
                <div className="grid grid-cols-7 md:grid-cols-10 gap-2 mb-4">
                  {Array.from({ length: games.find(g => g.id === selectedGameId)?.rules_json?.numbers?.max || 47 }, (_, i) => i + 1).map(num => (
                    <button
                      key={num}
                      onClick={() => toggleNumber(num)}
                      className={`p-2 rounded-lg text-sm font-medium transition-colors ${
                        selectedNumbers.includes(num)
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      {num}
                    </button>
                  ))}
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <p className="text-sm text-gray-600 mb-2">{t('probabilities.selectedNumbers')}</p>
                    <div className="flex flex-wrap gap-2">
                      {selectedNumbers.length > 0 ? (
                        selectedNumbers.map((num: any) => (
                          <span key={num} className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                            {num}
                          </span>
                        ))
                      ) : (
                        <span className="text-sm text-gray-400">{t('probabilities.noNumbersSelected')}</span>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={calculateCombinationProbability}
                    disabled={selectedNumbers.length === 0}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
                  >
                    {t('probabilities.calculate')}
                  </button>
                </div>
                
                {combinationProb !== null && (
                  <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-sm text-green-800 mb-1">{t('probabilities.combinationProb')}</p>
                    <p className="text-2xl font-bold text-green-900">
                      {(combinationProb * 100).toExponential(4)}%
                    </p>
                    <p className="text-xs text-green-700 mt-2">
                      {t('probabilities.odds')} {(1 / combinationProb).toExponential(2)}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {Object.entries(result.results.probabilities).map(([modelName, modelData]: [string, any]) => (
            <div key={modelName} className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                {modelName} - {modelData.method_id}
              </h2>

              {/* Top N Analytique */}
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-indigo-300 rounded-lg p-4 mb-6">
                <h4 className="font-semibold text-indigo-900 mb-3">🎯 Top 10 Numéros Analytiques (Classement par probabilité estimée)</h4>
                <div className="flex flex-wrap gap-2 mb-4">
                  {getTopNNumbers(modelData, 10).map((item, idx) => (
                    <div key={idx} className="flex flex-col items-center">
                      <span className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-indigo-600 text-white text-lg font-bold shadow-lg">
                        {item.number}
                      </span>
                      <span className="text-xs text-indigo-700 mt-1">P: {(item.probability * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-indigo-100">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">Rang</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">Numéro</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">p_est</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">p_uniforme</th>
                        <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">Δ</th>
                        {modelData.credible_interval_95 && (
                          <th className="px-3 py-2 text-left text-xs font-medium text-indigo-900">IC 95%</th>
                        )}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-indigo-200">
                      {getTopNNumbers(modelData, 10).map((item, idx) => (
                        <tr key={idx}>
                          <td className="px-3 py-2 whitespace-nowrap font-medium text-indigo-900">{idx + 1}</td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded font-medium">{item.number}</span>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">{item.probability.toFixed(4)}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-gray-600">{item.baseline.toFixed(4)}</td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className={item.delta > 0 ? 'text-green-600' : 'text-red-600'}>
                              {item.delta > 0 ? '+' : ''}{item.delta.toFixed(4)}
                            </span>
                          </td>
                          {modelData.credible_interval_95 && (
                            <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-600">
                              {item.credible_lower && item.credible_upper
                                ? `[${item.credible_lower.toFixed(3)}; ${item.credible_upper.toFixed(3)}]`
                                : '—'}
                            </td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {modelData.warnings && modelData.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-4 mb-4">
                  <h4 className="font-medium text-yellow-900 mb-2">{t('probabilities.warnings')}</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {modelData.warnings.map((warning: string, idx: number) => (
                      <li key={idx} className="text-yellow-800 text-sm">{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {modelData.evaluation && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="bg-gray-50 p-4 rounded">
                    <p className="text-sm text-gray-500">{t('probabilities.brierScore')}</p>
                    <p className="text-xl font-semibold">{modelData.evaluation.brier_score.toFixed(6)}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <p className="text-sm text-gray-500">{t('probabilities.baselineBrier')}</p>
                    <p className="text-xl font-semibold">{modelData.evaluation.baseline_brier.toFixed(6)}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <p className="text-sm text-gray-500">{t('probabilities.lift')}</p>
                    <p className={`text-xl font-semibold ${modelData.evaluation.lift > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {modelData.evaluation.lift.toFixed(6)}
                    </p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <p className="text-sm text-gray-500">{t('probabilities.ece')}</p>
                    <p className="text-xl font-semibold">{modelData.evaluation.ece.toFixed(6)}</p>
                  </div>
                </div>
              )}

              {modelData.top_numbers && (
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-2">{t('probabilities.topNumbers')}</h4>
                  <div className="flex flex-wrap gap-2">
                    {modelData.top_numbers.map((num: number) => (
                      <span key={num} className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                        {num}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <h4 className="font-medium text-gray-900 mb-4">{t('probabilities.probabilityDist')}</h4>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={getProbabilityChartData(modelData)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="number" />
                    <YAxis label={{ value: t('probabilities.probability'), angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="probability" fill="#3b82f6" name={t('probabilities.model')} />
                    <Bar dataKey="baseline" fill="#9ca3af" name={t('probabilities.baseline')} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Calibration Plot for M1 and M2 */}
              {(modelName === 'M1' || modelName === 'M2') && modelData.evaluation?.calibration_curve && (
                <div className="mt-6">
                  <h4 className="font-medium text-gray-900 mb-4">📊 Courbe de calibration (Reliability Diagram)</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={modelData.evaluation.calibration_curve.map((point: any, idx: number) => ({
                      predicted: point[0],
                      observed: point[1],
                      perfect: point[0]
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="predicted" label={{ value: 'Probabilité prédite', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Fréquence observée', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="observed" stroke="#3b82f6" name="Observé" strokeWidth={2} />
                      <Line type="monotone" dataKey="perfect" stroke="#9ca3af" strokeDasharray="5 5" name="Parfait" />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-600 mt-2">
                    ECE (Expected Calibration Error): {modelData.evaluation.ece.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
