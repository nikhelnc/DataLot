import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Shield, Play, RefreshCw, CheckCircle, XCircle, AlertTriangle, Info, ChevronDown, ChevronRight } from 'lucide-react'
import { api } from '../api/client'

interface Game {
  id: string
  name: string
  description: string
}

interface TestResult {
  statistic: number
  p_value: number
  passed: boolean
  description: string
  details?: Record<string, any>
}

interface ForensicsProfile {
  id: string
  game_id: string
  computed_at: string
  n_draws: number
  conformity_score: number
  conformity_ci_low: number
  conformity_ci_high: number
  conformity_interpretation: string
  generator_type: string
  category_scores: Record<string, number>
  nist_tests?: { tests: Record<string, TestResult>; summary: { n_passed: number; n_tests: number } }
  physical_tests?: { tests: Record<string, TestResult>; summary: { n_passed: number; n_tests: number } }
  rng_tests?: { tests: Record<string, TestResult>; summary: { n_passed: number; n_tests: number } }
  structural_tests?: { tests: Record<string, TestResult>; summary: { n_passed: number; n_tests: number } }
  computation_time_seconds?: number
}

function ConformityGauge({ score, ciLow, ciHigh }: { score: number; ciLow: number; ciHigh: number }) {
  const percentage = Math.round(score * 100)
  const isConforming = score >= ciLow
  const color = isConforming ? (score > ciHigh ? 'text-yellow-500' : 'text-green-500') : 'text-red-500'
  const bgColor = isConforming ? (score > ciHigh ? 'bg-yellow-500' : 'bg-green-500') : 'bg-red-500'
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-40 h-40">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="80"
            cy="80"
            r="70"
            stroke="#e5e7eb"
            strokeWidth="12"
            fill="none"
          />
          <circle
            cx="80"
            cy="80"
            r="70"
            stroke="currentColor"
            strokeWidth="12"
            fill="none"
            strokeDasharray={`${percentage * 4.4} 440`}
            className={color}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${color}`}>{percentage}%</span>
          <span className="text-xs text-gray-500">Conformité</span>
        </div>
      </div>
      <div className="mt-2 text-sm text-gray-600">
        IC 95% : [{Math.round(ciLow * 100)}% - {Math.round(ciHigh * 100)}%]
      </div>
    </div>
  )
}

function TestResultBadge({ passed, pValue }: { passed: boolean; pValue: number }) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
      passed ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
    }`}>
      {passed ? <CheckCircle className="w-3 h-3 mr-1" /> : <XCircle className="w-3 h-3 mr-1" />}
      p={pValue.toFixed(4)}
    </span>
  )
}

function TestCategory({ 
  title, 
  tests, 
  summary,
  defaultOpen = false 
}: { 
  title: string
  tests: Record<string, TestResult>
  summary: { n_passed: number; n_tests: number }
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
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
          <div className={`w-3 h-3 rounded-full ${passRate === 1 ? 'bg-green-500' : passRate >= 0.8 ? 'bg-yellow-500' : 'bg-red-500'}`} />
        </div>
      </button>
      
      {isOpen && (
        <div className="p-4 space-y-3">
          {Object.entries(tests).map(([name, result]) => (
            <div key={name} className="flex items-start justify-between p-3 bg-white border border-gray-100 rounded-lg">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-800">{name}</span>
                  <TestResultBadge passed={result.passed} pValue={result.p_value} />
                </div>
                <p className="text-sm text-gray-500 mt-1">{result.description}</p>
                {result.details && (
                  <div className="mt-2 text-xs text-gray-400">
                    Statistique: {result.statistic.toFixed(4)}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Forensics() {
  const { t } = useTranslation()
  const [games, setGames] = useState<Game[]>([])
  const [selectedGame, setSelectedGame] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [profile, setProfile] = useState<ForensicsProfile | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ step: string; progress: number; total: number } | null>(null)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGame) {
      loadLatestProfile()
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

  const loadLatestProfile = async () => {
    if (!selectedGame) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await api.get(`/games/${selectedGame}/forensics`)
      if (response.data) {
        setProfile(response.data)
      } else {
        setProfile(null)
      }
    } catch (err: any) {
      if (err.response?.status !== 404) {
        setError('Erreur lors du chargement du profil forensique')
      }
      setProfile(null)
    } finally {
      setLoading(false)
    }
  }

  const runForensics = async () => {
    if (!selectedGame) return
    
    setRunning(true)
    setError(null)
    setProgress({ step: 'Démarrage...', progress: 0, total: 100 })
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || '/api'
      const eventSource = new EventSource(
        `${apiUrl}/games/${selectedGame}/forensics/run/stream?n_simulations=1000&compute_ci=true&alpha=0.01`
      )
      
      eventSource.onmessage = async (event) => {
        const data = JSON.parse(event.data)
        
        if (data.error) {
          setError(data.error)
          eventSource.close()
          setRunning(false)
          setProgress(null)
        } else if (data.completed) {
          // Analysis completed, reload the profile
          const profileResponse = await api.get(`/games/${selectedGame}/forensics/${data.profile_id}`)
          setProfile(profileResponse.data)
          eventSource.close()
          setRunning(false)
          setProgress(null)
        } else if (data.step) {
          // Progress update
          setProgress({
            step: data.step,
            progress: data.progress,
            total: data.total
          })
        }
      }
      
      eventSource.onerror = () => {
        setError('Connexion perdue avec le serveur')
        eventSource.close()
        setRunning(false)
        setProgress(null)
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur lors de l\'analyse forensique')
      setRunning(false)
      setProgress(null)
    }
  }

  const getGeneratorTypeLabel = (type: string) => {
    switch (type) {
      case 'physical': return 'Machine physique'
      case 'rng': return 'Générateur logiciel (RNG)'
      case 'hybrid': return 'Hybride'
      default: return 'Inconnu'
    }
  }

  const getInterpretationColor = (interpretation: string) => {
    if (interpretation.includes('CONFORMING')) return 'text-green-600 bg-green-50 border-green-200'
    if (interpretation.includes('SUSPICIOUS')) return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
            <Shield className="w-8 h-8 text-blue-600" />
            Forensique du Générateur
          </h1>
          <p className="text-gray-500 mt-1">
            Analyse forensique complète pour évaluer la conformité du générateur
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
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {games.map((game) => (
                <option key={game.id} value={game.id}>
                  {game.name}
                </option>
              ))}
            </select>
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={runForensics}
              disabled={!selectedGame || running}
              className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
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
            
            <button
              onClick={loadLatestProfile}
              disabled={!selectedGame || loading}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Actualiser
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

      {/* Progress Bar */}
      {running && progress && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">{progress.step}</span>
            <span className="text-sm text-gray-500">{progress.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            Analyse forensique en cours... Veuillez patienter.
          </p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-500">Chargement du profil forensique...</p>
        </div>
      )}

      {/* No Profile */}
      {!loading && !profile && selectedGame && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <Shield className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Aucun profil forensique</h3>
          <p className="text-gray-500 mb-4">
            Lancez une analyse forensique pour évaluer la conformité du générateur.
          </p>
          <button
            onClick={runForensics}
            disabled={running}
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Play className="w-4 h-4 mr-2" />
            Lancer l'analyse
          </button>
        </div>
      )}

      {/* Profile Results */}
      {profile && (
        <div className="space-y-6">
          {/* Score Overview */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Conformity Gauge */}
              <div className="flex flex-col items-center justify-center">
                <ConformityGauge 
                  score={profile.conformity_score} 
                  ciLow={profile.conformity_ci_low}
                  ciHigh={profile.conformity_ci_high}
                />
              </div>
              
              {/* Interpretation */}
              <div className="md:col-span-2 space-y-4">
                <div className={`p-4 rounded-lg border ${getInterpretationColor(profile.conformity_interpretation || '')}`}>
                  <div className="flex items-start gap-3">
                    {profile.conformity_interpretation?.includes('CONFORMING') ? (
                      <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    ) : profile.conformity_interpretation?.includes('SUSPICIOUS') ? (
                      <AlertTriangle className="w-6 h-6 text-yellow-500 flex-shrink-0" />
                    ) : (
                      <XCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
                    )}
                    <div>
                      <h3 className="font-semibold">Interprétation</h3>
                      <p className="text-sm mt-1">{profile.conformity_interpretation}</p>
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-500">Type de générateur</div>
                    <div className="text-lg font-semibold text-gray-900">
                      {getGeneratorTypeLabel(profile.generator_type)}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="text-sm text-gray-500">Tirages analysés</div>
                    <div className="text-lg font-semibold text-gray-900">
                      {profile.n_draws}
                    </div>
                  </div>
                </div>
                
                {profile.computation_time_seconds && (
                  <div className="text-sm text-gray-400">
                    Calculé le {new Date(profile.computed_at).toLocaleString('fr-FR')} 
                    en {profile.computation_time_seconds.toFixed(1)}s
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Category Scores */}
          {profile.category_scores && Object.keys(profile.category_scores).length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Scores par catégorie</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(profile.category_scores).map(([category, score]) => (
                  <div key={category} className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                      {Math.round((score as number) * 100)}%
                    </div>
                    <div className="text-sm text-gray-500 mt-1">
                      {category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
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
              {profile.nist_tests?.tests && (
                <TestCategory 
                  title="Tests NIST (Randomness)" 
                  tests={profile.nist_tests.tests}
                  summary={profile.nist_tests.summary}
                />
              )}
              
              {profile.physical_tests?.tests && (
                <TestCategory 
                  title="Tests de biais physiques" 
                  tests={profile.physical_tests.tests}
                  summary={profile.physical_tests.summary}
                />
              )}
              
              {profile.rng_tests?.tests && (
                <TestCategory 
                  title="Tests RNG (Générateur logiciel)" 
                  tests={profile.rng_tests.tests}
                  summary={profile.rng_tests.summary}
                />
              )}
              
              {profile.structural_tests?.tests && (
                <TestCategory 
                  title="Tests structurels" 
                  tests={profile.structural_tests.tests}
                  summary={profile.structural_tests.summary}
                />
              )}
            </div>
          </div>

          {/* Scientific Warning */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium mb-1">Interprétation scientifique</p>
                <p>
                  Ces résultats évaluent la conformité statistique du générateur à un processus aléatoire uniforme.
                  Une déviation détectée ne prouve pas une fraude intentionnelle — elle peut résulter d'un défaut mécanique,
                  d'un biais de données, ou d'une coïncidence statistique.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
