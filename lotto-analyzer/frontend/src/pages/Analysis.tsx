import { useState, useEffect } from 'react'
import { gamesApi, analysesApi, drawsApi } from '../api/client'
import { Download, FileText, TrendingUp, AlertCircle, CheckCircle, BarChart3, FileJson } from 'lucide-react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { exportToCSV, exportToJSON, formatDrawsForExport, formatStatsForExport } from '../utils/exportData'

export default function Analysis() {
  const [games, setGames] = useState<any[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [analysisName, setAnalysisName] = useState<string>('full_analysis_v1')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [draws, setDraws] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGameId) {
      loadDrawsAndAnalyze()
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

  const loadDrawsAndAnalyze = async () => {
    if (!selectedGameId) return
    
    try {
      const response = await drawsApi.list(selectedGameId, { limit: 1000 })
      setDraws(response.data)
      calculateStats(response.data)
    } catch (error) {
      console.error('Failed to load draws:', error)
    }
  }

  const calculateStats = (drawsData: any[]) => {
    if (drawsData.length === 0) {
      setStats(null)
      return
    }

    const allNumbers = drawsData.flatMap(d => d.numbers)
    const freq: Record<number, number> = {}
    allNumbers.forEach(num => {
      freq[num] = (freq[num] || 0) + 1
    })

    const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1])
    const avgFreq = allNumbers.length / Object.keys(freq).length
    
    const consecutivePairs: Record<string, number> = {}
    drawsData.forEach(draw => {
      const nums = [...draw.numbers].sort((a, b) => a - b)
      for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i + 1] - nums[i] === 1) {
          const pair = `${nums[i]}-${nums[i + 1]}`
          consecutivePairs[pair] = (consecutivePairs[pair] || 0) + 1
        }
      }
    })

    const evenOddCounts = { even: 0, odd: 0 }
    allNumbers.forEach(num => {
      if (num % 2 === 0) evenOddCounts.even++
      else evenOddCounts.odd++
    })

    const sums = drawsData.map(d => d.numbers.reduce((a: number, b: number) => a + b, 0))
    const avgSum = sums.reduce((a, b) => a + b, 0) / sums.length

    setStats({
      totalDraws: drawsData.length,
      totalNumbers: allNumbers.length,
      uniqueNumbers: Object.keys(freq).length,
      frequency: freq,
      mostFrequent: sorted.slice(0, 10),
      leastFrequent: sorted.slice(-10).reverse(),
      avgFrequency: avgFreq.toFixed(2),
      consecutivePairs: Object.entries(consecutivePairs).sort((a, b) => b[1] - a[1]).slice(0, 10),
      evenOdd: evenOddCounts,
      avgSum: avgSum.toFixed(2),
      minSum: Math.min(...sums),
      maxSum: Math.max(...sums)
    })
  }

  const handleRunAnalysis = async () => {
    if (!selectedGameId) return

    setLoading(true)
    try {
      const response = await analysesApi.run({
        game_id: selectedGameId,
        analysis_name: analysisName,
        params: {},
      })
      setResult(response.data)
    } catch (error: any) {
      console.error('Analysis failed:', error)
      setResult({ error: error.response?.data?.detail || 'Analysis failed' })
    } finally {
      setLoading(false)
    }
  }

  const handleExportCsv = async () => {
    if (!result?.analysis_id) return
    try {
      const response = await analysesApi.exportCsv(result.analysis_id)
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `analysis_${result.analysis_id}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const handleViewReport = async () => {
    if (!result?.analysis_id) return
    try {
      const response = await analysesApi.getReport(result.analysis_id)
      const newWindow = window.open()
      if (newWindow) {
        newWindow.document.write(response.data)
        newWindow.document.close()
      }
    } catch (error) {
      console.error('Report failed:', error)
    }
  }

  const handleExportStats = () => {
    if (!stats) return
    const formattedStats = formatStatsForExport(stats)
    exportToJSON(formattedStats, `lotto-stats-${new Date().toISOString().split('T')[0]}.json`)
  }

  const handleExportDraws = () => {
    if (draws.length === 0) return
    const formattedDraws = formatDrawsForExport(draws)
    exportToCSV(formattedDraws, `lotto-draws-${new Date().toISOString().split('T')[0]}.csv`)
  }

  return (
    <div className="px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Run Analysis</h1>
        {stats && (
          <div className="flex space-x-2">
            <button
              onClick={handleExportStats}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <FileJson className="w-4 h-4 mr-2" />
              Export Stats (JSON)
            </button>
            <button
              onClick={handleExportDraws}
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <Download className="w-4 h-4 mr-2" />
              Export Draws (CSV)
            </button>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Game
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

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Analysis Type
            </label>
            <select
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              value={analysisName}
              onChange={(e) => setAnalysisName(e.target.value)}
            >
              <option value="descriptive_v1">Descriptive Statistics</option>
              <option value="randomness_tests_v1">Randomness Tests</option>
              <option value="anomaly_detection_v1">Anomaly Detection</option>
              <option value="forecast_probabilities_v1">Probability Forecast</option>
              <option value="full_analysis_v1">Full Analysis</option>
            </select>
          </div>

          <button
            onClick={handleRunAnalysis}
            disabled={!selectedGameId || loading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
          >
            {loading ? 'Running...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {stats && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-500">Total Draws</p>
                  <p className="text-3xl font-bold text-gray-900 mt-2">{stats.totalDraws}</p>
                </div>
                <BarChart3 className="w-10 h-10 text-blue-500" />
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-500">Unique Numbers</p>
                  <p className="text-3xl font-bold text-gray-900 mt-2">{stats.uniqueNumbers}</p>
                  <p className="text-xs text-gray-500 mt-1">Avg freq: {stats.avgFrequency}</p>
                </div>
                <TrendingUp className="w-10 h-10 text-green-500" />
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-500">Average Sum</p>
                  <p className="text-3xl font-bold text-gray-900 mt-2">{stats.avgSum}</p>
                  <p className="text-xs text-gray-500 mt-1">Range: {stats.minSum} - {stats.maxSum}</p>
                </div>
                <AlertCircle className="w-10 h-10 text-purple-500" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Number Frequency Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stats.mostFrequent.map(([num, count]: [string, number]) => ({ number: num, frequency: count }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="number" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="frequency" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-4 grid grid-cols-5 gap-2">
                {stats.mostFrequent.slice(0, 10).map(([num, count]: [string, number], idx: number) => (
                  <div key={idx} className="text-center p-2 bg-blue-50 rounded">
                    <div className="text-lg font-bold text-blue-600">{num}</div>
                    <div className="text-xs text-gray-600">{count}x</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Even vs Odd Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Even', value: stats.evenOdd.even },
                      { name: 'Odd', value: stats.evenOdd.odd }
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    <Cell fill="#3b82f6" />
                    <Cell fill="#f59e0b" />
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded">
                  <div className="text-2xl font-bold text-blue-600">{stats.evenOdd.even}</div>
                  <div className="text-sm text-gray-600">Even Numbers</div>
                </div>
                <div className="text-center p-3 bg-orange-50 rounded">
                  <div className="text-2xl font-bold text-orange-600">{stats.evenOdd.odd}</div>
                  <div className="text-sm text-gray-600">Odd Numbers</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Consecutive Number Pairs</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {stats.consecutivePairs.map(([pair, count]: [string, number], idx: number) => (
                <div key={idx} className="p-3 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg text-center">
                  <div className="text-lg font-bold text-purple-600">{pair}</div>
                  <div className="text-xs text-gray-600">{count} times</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Least Frequent Numbers</h3>
            <div className="grid grid-cols-5 md:grid-cols-10 gap-2">
              {stats.leastFrequent.map(([num, count]: [string, number], idx: number) => (
                <div key={idx} className="text-center p-2 bg-red-50 rounded">
                  <div className="text-lg font-bold text-red-600">{num}</div>
                  <div className="text-xs text-gray-600">{count}x</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Advanced Analysis Results</h2>
            {result.analysis_id && (
              <div className="flex space-x-2">
                <button
                  onClick={handleExportCsv}
                  className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export CSV
                </button>
                <button
                  onClick={handleViewReport}
                  className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  View Report
                </button>
              </div>
            )}
          </div>

          {result.error ? (
            <div className="bg-red-50 border border-red-200 rounded p-4">
              <p className="text-red-800">{result.error}</p>
            </div>
          ) : (
            <div className="space-y-6">
              {result.results?.summary && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Summary</h3>
                  <p className="text-gray-700">{result.results.summary}</p>
                </div>
              )}

              {result.results?.warnings && result.results.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded p-4">
                  <h3 className="text-lg font-medium text-yellow-900 mb-2">⚠️ Warnings</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {result.results.warnings.map((warning: string, idx: number) => (
                      <li key={idx} className="text-yellow-800">{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.results?.metrics && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Metrics</h3>
                  <pre className="bg-gray-50 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(result.results.metrics, null, 2)}
                  </pre>
                </div>
              )}

              {result.results?.tests && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Statistical Tests</h3>
                  <pre className="bg-gray-50 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(result.results.tests, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
