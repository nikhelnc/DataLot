import { useState, useEffect } from 'react'
import { alertsApi, gamesApi, Alert, Game } from '../api/client'
import { AlertTriangle } from 'lucide-react'

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [games, setGames] = useState<Game[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [severity, setSeverity] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    loadAlerts()
  }, [selectedGameId, severity])

  const loadGames = async () => {
    try {
      const response = await gamesApi.list()
      setGames(response.data)
    } catch (error) {
      console.error('Failed to load games:', error)
    }
  }

  const loadAlerts = async () => {
    setLoading(true)
    try {
      const params: any = {}
      if (selectedGameId) params.game_id = selectedGameId
      if (severity) params.severity = severity

      const response = await alertsApi.list(params)
      setAlerts(response.data)
    } catch (error) {
      console.error('Failed to load alerts:', error)
    } finally {
      setLoading(false)
    }
  }

  const getSeverityColor = (sev: string) => {
    switch (sev) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200'
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'low': return 'bg-blue-100 text-blue-800 border-blue-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  return (
    <div className="px-4 py-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Alerts</h1>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Game
            </label>
            <select
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={selectedGameId}
              onChange={(e) => setSelectedGameId(e.target.value)}
            >
              <option value="">All Games</option>
              {games.map(game => (
                <option key={game.id} value={game.id}>
                  {game.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter by Severity
            </label>
            <select
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={severity}
              onChange={(e) => setSeverity(e.target.value)}
            >
              <option value="">All Severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">Loading...</div>
      ) : alerts.length === 0 ? (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <p className="text-green-800">No alerts found. This is good news!</p>
        </div>
      ) : (
        <div className="space-y-4">
          {alerts.map(alert => (
            <div
              key={alert.id}
              className={`border rounded-lg p-6 ${getSeverityColor(alert.severity)}`}
            >
              <div className="flex items-start">
                <AlertTriangle className="w-6 h-6 mr-3 flex-shrink-0" />
                <div className="flex-1">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-lg font-semibold">{alert.message}</h3>
                    <span className="text-sm font-medium uppercase">{alert.severity}</span>
                  </div>
                  <p className="text-sm mb-2">
                    Score: {alert.score} | {new Date(alert.created_at).toLocaleString()}
                  </p>
                  {alert.evidence_json && (
                    <details className="mt-4">
                      <summary className="cursor-pointer text-sm font-medium">
                        View Evidence
                      </summary>
                      <pre className="mt-2 bg-white bg-opacity-50 p-3 rounded text-xs overflow-x-auto">
                        {JSON.stringify(alert.evidence_json, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
