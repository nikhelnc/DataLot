import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { DollarSign, Play, RefreshCw, TrendingUp, CheckCircle, XCircle, ChevronDown, ChevronRight } from 'lucide-react'
import { api } from '../api/client'

interface Game {
  id: string
  name: string
}

interface JackpotStats {
  has_jackpot_data: boolean
  n_draws_with_jackpot?: number
  jackpot_stats?: {
    min: number
    max: number
    mean: number
    median: number
  }
  rollover_stats?: {
    n_rollovers: number
    rollover_rate: number
  }
  must_be_won_stats?: {
    n_mbw: number
    mbw_rate: number
  }
}

interface AnalysisResult {
  analysis_id: string
  n_draws: number
  n_draws_with_jackpot: number
  jackpot_range: { min: number; max: number; mean: number }
  independence_tests: any
  player_bias: any
  rdd_analysis: any
  must_be_won: any
}

function formatCurrency(value: number): string {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M €`
  } else if (value >= 1000) {
    return `${(value / 1000).toFixed(0)}K €`
  }
  return `${value.toFixed(0)} €`
}

function TestResultItem({ name, result }: { name: string; result: any }) {
  const passed = result.passed !== false && result.p_value >= 0.05
  
  return (
    <div className="flex items-start justify-between p-3 bg-white border border-gray-100 rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          {passed ? (
            <CheckCircle className="w-4 h-4 text-green-500" />
          ) : (
            <XCircle className="w-4 h-4 text-red-500" />
          )}
          <span className="font-medium text-gray-800">{name}</span>
        </div>
        <p className="text-sm text-gray-500 mt-1">{result.description}</p>
        {result.p_value !== undefined && (
          <div className="mt-1 text-xs text-gray-400">
            p-value: {result.p_value.toFixed(4)}
            {result.statistic !== undefined && ` | Stat: ${result.statistic.toFixed(4)}`}
          </div>
        )}
      </div>
    </div>
  )
}

function AnalysisSection({ 
  title, 
  data,
  defaultOpen = false 
}: { 
  title: string
  data: any
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  
  if (!data) return null
  
  const tests = data.tests || data.analyses || {}
  const summary = data.summary || {}
  
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
        {summary.conclusion && (
          <span className={`text-sm px-2 py-1 rounded ${
            summary.conclusion === 'INDEPENDENT' || summary.conclusion === 'NO_DIFFERENCE' 
              ? 'bg-green-100 text-green-700' 
              : 'bg-yellow-100 text-yellow-700'
          }`}>
            {summary.conclusion}
          </span>
        )}
      </button>
      
      {isOpen && (
        <div className="p-4 space-y-3">
          {Object.entries(tests).map(([name, result]: [string, any]) => (
            <TestResultItem key={name} name={name} result={result} />
          ))}
        </div>
      )}
    </div>
  )
}

export default function Jackpot() {
  const { t } = useTranslation()
  const [games, setGames] = useState<Game[]>([])
  const [selectedGame, setSelectedGame] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [stats, setStats] = useState<JackpotStats | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGame) {
      loadStats()
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

  const loadStats = async () => {
    if (!selectedGame) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get(`/games/${selectedGame}/jackpot/stats`)
      setStats(response.data)
    } catch (err: any) {
      if (err.response?.status !== 404) {
        setError('Erreur lors du chargement des statistiques')
      }
    } finally {
      setLoading(false)
    }
  }

  const runAnalysis = async () => {
    if (!selectedGame) return
    
    setRunning(true)
    setError(null)
    
    try {
      const response = await api.post(`/games/${selectedGame}/jackpot/analyze`, {
        alpha: 0.05
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur lors de l\'analyse')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
            <DollarSign className="w-8 h-8 text-green-600" />
            Analyse Jackpot
          </h1>
          <p className="text-gray-500 mt-1">
            Analyse de l'indépendance entre jackpot et résultats des tirages
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
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
            >
              {games.map((game) => (
                <option key={game.id} value={game.id}>{game.name}</option>
              ))}
            </select>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={runAnalysis}
              disabled={!selectedGame || running || !stats?.has_jackpot_data}
              className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
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
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* No Jackpot Data */}
      {stats && !stats.has_jackpot_data && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <DollarSign className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-yellow-800 mb-2">Aucune donnée jackpot</h3>
          <p className="text-yellow-700">
            Ce jeu ne contient pas de données de jackpot. Importez des tirages avec les colonnes jackpot pour activer cette analyse.
          </p>
        </div>
      )}

      {/* Jackpot Stats */}
      {stats?.has_jackpot_data && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            Statistiques Jackpot
          </h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-green-50 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-700">
                {stats.n_draws_with_jackpot}
              </div>
              <div className="text-sm text-green-600">Tirages avec jackpot</div>
            </div>
            
            {stats.jackpot_stats && (
              <>
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-xl font-bold text-gray-700">
                    {formatCurrency(stats.jackpot_stats.min)}
                  </div>
                  <div className="text-sm text-gray-500">Minimum</div>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-xl font-bold text-gray-700">
                    {formatCurrency(stats.jackpot_stats.max)}
                  </div>
                  <div className="text-sm text-gray-500">Maximum</div>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg text-center">
                  <div className="text-xl font-bold text-gray-700">
                    {formatCurrency(stats.jackpot_stats.mean)}
                  </div>
                  <div className="text-sm text-gray-500">Moyenne</div>
                </div>
              </>
            )}
          </div>
          
          {(stats.rollover_stats || stats.must_be_won_stats) && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              {stats.rollover_stats && (
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-lg font-semibold text-blue-700">
                    {stats.rollover_stats.n_rollovers} rollovers
                  </div>
                  <div className="text-sm text-blue-600">
                    Taux: {(stats.rollover_stats.rollover_rate * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              
              {stats.must_be_won_stats && (
                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="text-lg font-semibold text-purple-700">
                    {stats.must_be_won_stats.n_mbw} "Must Be Won"
                  </div>
                  <div className="text-sm text-purple-600">
                    Taux: {(stats.must_be_won_stats.mbw_rate * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Analysis Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Résumé de l'analyse</h2>
            
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-700">{result.n_draws}</div>
                <div className="text-sm text-gray-500">Tirages analysés</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-700">{result.n_draws_with_jackpot}</div>
                <div className="text-sm text-gray-500">Avec données jackpot</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-xl font-bold text-gray-700">
                  {formatCurrency(result.jackpot_range.min)} - {formatCurrency(result.jackpot_range.max)}
                </div>
                <div className="text-sm text-gray-500">Plage jackpot</div>
              </div>
            </div>
            
            {/* Independence conclusion */}
            {result.independence_tests?.summary && (
              <div className={`p-4 rounded-lg border ${
                result.independence_tests.summary.conclusion === 'INDEPENDENT'
                  ? 'bg-green-50 border-green-200'
                  : 'bg-yellow-50 border-yellow-200'
              }`}>
                <p className={`font-medium ${
                  result.independence_tests.summary.conclusion === 'INDEPENDENT'
                    ? 'text-green-800'
                    : 'text-yellow-800'
                }`}>
                  {result.independence_tests.summary.conclusion === 'INDEPENDENT'
                    ? '✓ Les tirages sont indépendants du montant du jackpot'
                    : '⚠ Dépendance potentielle détectée entre jackpot et tirages'}
                </p>
                <p className="text-sm mt-1 text-gray-600">
                  {result.independence_tests.summary.n_passed}/{result.independence_tests.summary.n_tests} tests passés
                </p>
              </div>
            )}
          </div>

          {/* Detailed Results */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Résultats détaillés</h2>
            <div className="space-y-4">
              <AnalysisSection 
                title="Tests d'indépendance" 
                data={result.independence_tests}
                defaultOpen={true}
              />
              
              <AnalysisSection 
                title="Analyse des biais joueurs" 
                data={result.player_bias}
              />
              
              {result.must_be_won && result.must_be_won.n_mbw_draws > 0 && (
                <AnalysisSection 
                  title="Analyse Must-Be-Won" 
                  data={result.must_be_won}
                />
              )}
              
              {result.rdd_analysis && !result.rdd_analysis.error && (
                <AnalysisSection 
                  title="Analyse RDD (seuils)" 
                  data={{
                    tests: result.rdd_analysis.threshold_results,
                    summary: { conclusion: result.rdd_analysis.optimal_threshold?.significant ? 'DISCONTINUITY' : 'NO_DISCONTINUITY' }
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* No Results */}
      {!result && !loading && stats?.has_jackpot_data && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <DollarSign className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Aucune analyse</h3>
          <p className="text-gray-500 mb-4">
            Lancez une analyse pour vérifier l'indépendance entre jackpot et tirages.
          </p>
          <button
            onClick={runAnalysis}
            disabled={running}
            className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            <Play className="w-4 h-4 mr-2" />
            Lancer l'analyse
          </button>
        </div>
      )}
    </div>
  )
}
