import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { AlertTriangle, Play, RefreshCw, CheckCircle, XCircle, Shield, Eye, ChevronDown, ChevronRight } from 'lucide-react'
import { api } from '../api/client'

interface Game {
  id: string
  name: string
}

interface Alert {
  id: string
  severity: string
  signal_type: string
  category: string
  title: string
  description: string
  status: string
  created_at: string
  statistical_evidence?: {
    p_value?: number
    statistic?: number
  }
}

interface FraudAnalysisResult {
  game_id: string
  n_draws: number
  period: { start: string; end: string }
  fraud_score: {
    score: number
    risk_level: string
    interpretation: string
    category_scores: Record<string, number>
  }
  alerts: {
    total: number
    by_severity: Record<string, number>
    details: any[]
  }
  test_results: {
    dispersion: any
    benford: any
    clustering: any
    jackpot: any
  }
}

function RiskGauge({ score, riskLevel }: { score: number; riskLevel: string }) {
  const getColor = () => {
    switch (riskLevel) {
      case 'CRITICAL': return 'text-red-600'
      case 'HIGH': return 'text-orange-500'
      case 'MEDIUM': return 'text-yellow-500'
      default: return 'text-green-500'
    }
  }
  
  const getBgColor = () => {
    switch (riskLevel) {
      case 'CRITICAL': return 'bg-red-500'
      case 'HIGH': return 'bg-orange-500'
      case 'MEDIUM': return 'bg-yellow-500'
      default: return 'bg-green-500'
    }
  }

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-40 h-40">
        <svg className="w-full h-full transform -rotate-90">
          <circle cx="80" cy="80" r="70" stroke="#e5e7eb" strokeWidth="12" fill="none" />
          <circle
            cx="80" cy="80" r="70"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            strokeDasharray={`${score * 4.4} 440`}
            className={getColor()}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${getColor()}`}>{Math.round(score)}</span>
          <span className="text-xs text-gray-500">Score de risque</span>
        </div>
      </div>
      <div className={`mt-2 px-3 py-1 rounded-full text-sm font-medium text-white ${getBgColor()}`}>
        {riskLevel}
      </div>
    </div>
  )
}

function SeverityBadge({ severity }: { severity: string }) {
  const colors: Record<string, string> = {
    CRITICAL: 'bg-red-100 text-red-800 border-red-200',
    HIGH: 'bg-orange-100 text-orange-800 border-orange-200',
    WARNING: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    INFO: 'bg-blue-100 text-blue-800 border-blue-200'
  }
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${colors[severity] || colors.INFO}`}>
      {severity}
    </span>
  )
}

function TestCategory({ 
  title, 
  results,
  defaultOpen = false 
}: { 
  title: string
  results: any
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  
  if (!results || !results.tests) return null
  
  const summary = results.summary || {}
  const passRate = summary.n_tests > 0 ? summary.n_passed / summary.n_tests : 1
  
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 bg-gray-50 flex items-center justify-between hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          {isOpen ? <ChevronDown className="w-5 h-5 text-gray-500" /> : <ChevronRight className="w-5 h-5 text-gray-500" />}
          <span className="font-medium text-gray-900">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm ${passRate === 1 ? 'text-green-600' : passRate >= 0.8 ? 'text-yellow-600' : 'text-red-600'}`}>
            {summary.n_passed}/{summary.n_tests} tests passés
          </span>
          {summary.max_severity && summary.max_severity !== 'INFO' && (
            <SeverityBadge severity={summary.max_severity} />
          )}
        </div>
      </button>
      
      {isOpen && (
        <div className="p-4 space-y-3">
          {Object.entries(results.tests).map(([name, result]: [string, any]) => (
            <div key={name} className="flex items-start justify-between p-3 bg-white border border-gray-100 rounded-lg">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {result.passed ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-500" />
                  )}
                  <span className="font-medium text-gray-800">{name}</span>
                  <SeverityBadge severity={result.severity || 'INFO'} />
                </div>
                <p className="text-sm text-gray-500 mt-1">{result.description}</p>
                <div className="mt-2 text-xs text-gray-400">
                  p-value: {result.p_value?.toFixed(4)} | Statistique: {result.statistic?.toFixed(4)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Fraud() {
  const { t } = useTranslation()
  const [games, setGames] = useState<Game[]>([])
  const [selectedGame, setSelectedGame] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<FraudAnalysisResult | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGame) {
      loadAlerts()
    }
  }, [selectedGame])

  const loadGames = async () => {
    try {
      const response = await api.get('/games')
      setGames(response.data)
      if (response.data.length > 0) {
        setSelectedGame(response.data[0].id)
      }
    } catch (err) {
      console.error('Error loading games:', err)
    }
  }

  const loadAlerts = async () => {
    if (!selectedGame) return
    
    setLoading(true)
    try {
      const response = await api.get(`/games/${selectedGame}/fraud/alerts?status=OPEN&limit=20`)
      setAlerts(response.data.alerts || [])
    } catch (err) {
      console.error('Error loading alerts:', err)
    } finally {
      setLoading(false)
    }
  }

  const runAnalysis = async () => {
    if (!selectedGame) return
    
    setRunning(true)
    setError(null)
    
    try {
      const response = await api.post(`/games/${selectedGame}/fraud/analyze`, {
        include_jackpot_tests: true,
        alpha: 0.01
      })
      setResult(response.data)
      loadAlerts()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur lors de l\'analyse')
    } finally {
      setRunning(false)
    }
  }

  const updateAlertStatus = async (alertId: string, status: string) => {
    try {
      await api.patch(`/games/${selectedGame}/fraud/alerts/${alertId}`, { status })
      loadAlerts()
    } catch (err) {
      console.error('Error updating alert:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
            <AlertTriangle className="w-8 h-8 text-orange-500" />
            Détection de Fraude
          </h1>
          <p className="text-gray-500 mt-1">
            Analyse statistique pour détecter les anomalies et signaux de fraude potentielle
          </p>
        </div>
      </div>

      {/* Game Selection & Actions */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sélectionner un jeu
            </label>
            <select
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
            >
              {games.map((game) => (
                <option key={game.id} value={game.id}>{game.name}</option>
              ))}
            </select>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={runAnalysis}
              disabled={!selectedGame || running}
              className="inline-flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50"
            >
              {running ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Analyse en cours...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Lancer l'analyse
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-medium text-red-800">Erreur</h3>
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Open Alerts */}
      {alerts.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Shield className="w-5 h-5 text-orange-500" />
            Alertes ouvertes ({alerts.length})
          </h2>
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div key={alert.id} className="flex items-start justify-between p-4 bg-gray-50 rounded-lg border border-gray-100">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <SeverityBadge severity={alert.severity} />
                    <span className="font-medium text-gray-900">{alert.title}</span>
                  </div>
                  <p className="text-sm text-gray-600">{alert.description}</p>
                  <div className="text-xs text-gray-400 mt-1">
                    {new Date(alert.created_at).toLocaleString('fr-FR')}
                    {alert.statistical_evidence?.p_value && (
                      <span className="ml-2">p-value: {alert.statistical_evidence.p_value.toFixed(4)}</span>
                    )}
                  </div>
                </div>
                <div className="flex gap-2 ml-4">
                  <button
                    onClick={() => updateAlertStatus(alert.id, 'INVESTIGATING')}
                    className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                  >
                    <Eye className="w-3 h-3 inline mr-1" />
                    Investiguer
                  </button>
                  <button
                    onClick={() => updateAlertStatus(alert.id, 'FALSE_POSITIVE')}
                    className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                  >
                    Faux positif
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {result && (
        <div className="space-y-6">
          {/* Score Overview */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Risk Gauge */}
              <div className="flex flex-col items-center justify-center">
                <RiskGauge score={result.fraud_score.score} riskLevel={result.fraud_score.risk_level} />
              </div>
              
              {/* Interpretation */}
              <div className="md:col-span-2 space-y-4">
                <div className={`p-4 rounded-lg border ${
                  result.fraud_score.risk_level === 'CRITICAL' ? 'bg-red-50 border-red-200 text-red-800' :
                  result.fraud_score.risk_level === 'HIGH' ? 'bg-orange-50 border-orange-200 text-orange-800' :
                  result.fraud_score.risk_level === 'MEDIUM' ? 'bg-yellow-50 border-yellow-200 text-yellow-800' :
                  'bg-green-50 border-green-200 text-green-800'
                }`}>
                  <p className="font-medium">{result.fraud_score.interpretation}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-500">Tirages analysés</div>
                    <div className="text-lg font-semibold text-gray-900">{result.n_draws}</div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-500">Alertes générées</div>
                    <div className="text-lg font-semibold text-gray-900">{result.alerts.total}</div>
                  </div>
                </div>
                
                {/* Alerts by severity */}
                <div className="flex gap-2">
                  {Object.entries(result.alerts.by_severity).map(([sev, count]) => (
                    count > 0 && (
                      <div key={sev} className="flex items-center gap-1">
                        <SeverityBadge severity={sev} />
                        <span className="text-sm text-gray-600">×{count}</span>
                      </div>
                    )
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Category Scores */}
          {result.fraud_score.category_scores && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Scores par catégorie</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(result.fraud_score.category_scores).map(([category, score]) => (
                  <div key={category} className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className={`text-2xl font-bold ${
                      (score as number) >= 50 ? 'text-red-600' :
                      (score as number) >= 25 ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {Math.round(score as number)}
                    </div>
                    <div className="text-sm text-gray-500 mt-1 capitalize">
                      {category}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Test Results */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Résultats détaillés des tests</h2>
            <div className="space-y-4">
              <TestCategory title="Tests de Dispersion" results={result.test_results.dispersion} />
              <TestCategory title="Tests de Benford" results={result.test_results.benford} />
              <TestCategory title="Tests de Clustering" results={result.test_results.clustering} />
              {result.test_results.jackpot && (
                <TestCategory title="Tests Jackpot" results={result.test_results.jackpot} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* No Results */}
      {!result && !loading && selectedGame && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <AlertTriangle className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Aucune analyse</h3>
          <p className="text-gray-500 mb-4">
            Lancez une analyse de fraude pour détecter les anomalies statistiques.
          </p>
          <button
            onClick={runAnalysis}
            disabled={running}
            className="inline-flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600"
          >
            <Play className="w-4 h-4 mr-2" />
            Lancer l'analyse
          </button>
        </div>
      )}
    </div>
  )
}
