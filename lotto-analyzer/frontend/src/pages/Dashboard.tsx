import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { gamesApi, drawsApi } from '../api/client'
import QuickActions from '../components/QuickActions'

export default function Dashboard() {
  const [games, setGames] = useState<any[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [draws, setDraws] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

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
    
    setLoading(true)
    try {
      const response = await drawsApi.list(selectedGameId, { limit: 1000 })
      setDraws(response.data)
    } catch (error) {
      console.error('Failed to load draws:', error)
    } finally {
      setLoading(false)
    }
  }

  const frequencyData = () => {
    const freq: Record<number, number> = {}
    draws.forEach(draw => {
      draw.numbers.forEach(num => {
        freq[num] = (freq[num] || 0) + 1
      })
    })
    return Object.entries(freq)
      .map(([num, count]) => ({ number: num, count }))
      .sort((a, b) => parseInt(a.number) - parseInt(b.number))
  }

  const getStats = () => {
    if (draws.length === 0) return null

    const allNumbers = draws.flatMap(d => d.numbers)
    const freq: Record<number, number> = {}
    allNumbers.forEach(num => {
      freq[num] = (freq[num] || 0) + 1
    })

    const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1])
    const mostFrequent = sorted.slice(0, 5).map(([num]) => parseInt(num))
    const leastFrequent = sorted.slice(-5).reverse().map(([num]) => parseInt(num))

    const dates = draws.map(d => new Date(d.draw_date)).sort((a, b) => a.getTime() - b.getTime())
    const dateRange = dates.length > 0 
      ? `${dates[0].toLocaleDateString('fr-FR')} - ${dates[dates.length - 1].toLocaleDateString('fr-FR')}`
      : ''

    const sums = draws.map(d => d.numbers.reduce((a, b) => a + b, 0))
    const avgSum = sums.reduce((a, b) => a + b, 0) / sums.length
    const minSum = Math.min(...sums)
    const maxSum = Math.max(...sums)

    return {
      totalDraws: draws.length,
      dateRange,
      mostFrequent,
      leastFrequent,
      avgSum: avgSum.toFixed(1),
      minSum,
      maxSum
    }
  }

  const stats = getStats()

  return (
    <div className="px-4 py-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Dashboard</h1>

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
        </div>
      </div>

      <QuickActions />

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium">Total Draws</p>
                <p className="text-3xl font-bold mt-2">{stats.totalDraws}</p>
              </div>
              <div className="bg-blue-400 bg-opacity-30 rounded-full p-3">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm font-medium">Average Sum</p>
                <p className="text-3xl font-bold mt-2">{stats.avgSum}</p>
                <p className="text-green-100 text-xs mt-1">Range: {stats.minSum} - {stats.maxSum}</p>
              </div>
              <div className="bg-green-400 bg-opacity-30 rounded-full p-3">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
            <div>
              <p className="text-purple-100 text-sm font-medium mb-2">Most Frequent</p>
              <div className="flex flex-wrap gap-2">
                {stats.mostFrequent.slice(0, 5).map((num, idx) => (
                  <span key={idx} className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-white bg-opacity-20 text-white font-bold text-sm">
                    {num}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg shadow-lg p-6 text-white">
            <div>
              <p className="text-orange-100 text-sm font-medium mb-2">Least Frequent</p>
              <div className="flex flex-wrap gap-2">
                {stats.leastFrequent.slice(0, 5).map((num, idx) => (
                  <span key={idx} className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-white bg-opacity-20 text-white font-bold text-sm">
                    {num}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Loading draws...</p>
        </div>
      ) : draws.length > 0 ? (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Number Frequency</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={frequencyData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="number" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : selectedGameId ? (
        <div className="bg-gray-50 rounded-lg p-12 text-center">
          <p className="text-gray-600">No draws found for this game. Import some data to get started.</p>
        </div>
      ) : null}
    </div>
  )
}
