import { useState, useEffect } from 'react'
import { Upload } from 'lucide-react'
import { gamesApi, importsApi, Game } from '../api/client'

export default function Import() {
  const [games, setGames] = useState<Game[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [previewDone, setPreviewDone] = useState(false)

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
      setPreviewDone(false)
    }
  }

  const handlePreview = async () => {
    if (!file || !selectedGameId) return

    setLoading(true)
    try {
      const response = await importsApi.upload(selectedGameId, file, 'preview')
      setResult(response.data)
      setPreviewDone(true)
    } catch (error: any) {
      console.error('Preview failed:', error)
      setResult({ error: error.response?.data?.detail || 'Preview failed' })
      setPreviewDone(false)
    } finally {
      setLoading(false)
    }
  }

  const handleCommit = async () => {
    if (!file || !selectedGameId) return

    setLoading(true)
    try {
      const response = await importsApi.upload(selectedGameId, file, 'commit')
      setResult(response.data)
      setFile(null)
      setPreviewDone(false)
    } catch (error: any) {
      console.error('Commit failed:', error)
      setResult({ error: error.response?.data?.detail || 'Commit failed' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="px-4 py-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Import Draws</h1>

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
            {/* Game configuration display */}
            {selectedGameId && games.find(g => g.id === selectedGameId) && (() => {
              const game = games.find(g => g.id === selectedGameId);
              const rules = game?.rules_json;
              if (!rules) return null;
              const mainDrawn = rules.main?.drawn || rules.main?.pick || 7;
              const bonusDrawn = rules.bonus?.drawn || rules.bonus?.pick || 0;
              return (
                <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded text-sm">
                  <div className="font-medium text-blue-900 mb-2">Game Configuration:</div>
                  <div className="text-blue-800">
                    <div><strong>Main:</strong> Pick {rules.main?.pick || '?'}, Draw {mainDrawn} from {rules.main?.min || 1}-{rules.main?.max || 49}</div>
                    {rules.bonus?.enabled && (
                      <div><strong>Bonus:</strong> Pick {rules.bonus.pick || 0}, Draw {bonusDrawn} {rules.bonus.separate_pool ? `from ${rules.bonus.min}-${rules.bonus.max}` : '(same pool)'}</div>
                    )}
                  </div>
                  <div className="mt-2 text-blue-700 text-xs">
                    <strong>Expected CSV format:</strong> draw_number;draw_date;n1;n2;...;n{mainDrawn}{bonusDrawn > 0 ? `;bonus1;...;bonus${bonusDrawn}` : ''}
                  </div>
                </div>
              );
            })()}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              CSV File
            </label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
              <div className="space-y-1 text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="flex text-sm text-gray-600">
                  <label className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500">
                    <span>Upload a file</span>
                    <input
                      type="file"
                      className="sr-only"
                      accept=".csv"
                      onChange={handleFileChange}
                    />
                  </label>
                </div>
                <p className="text-xs text-gray-500">CSV with semicolon separator (;)</p>
                {file && <p className="text-sm text-gray-700 mt-2">Selected: {file.name}</p>}
              </div>
            </div>
          </div>

          <div className="flex space-x-4">
            <button
              onClick={handlePreview}
              disabled={!file || loading || previewDone}
              className="flex-1 flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
            >
              {loading ? 'Processing...' : 'Preview'}
            </button>
            {previewDone && (
              <button
                onClick={handleCommit}
                disabled={loading}
                className="flex-1 flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400"
              >
                {loading ? 'Committing...' : 'Commit Import'}
              </button>
            )}
          </div>
        </div>
      </div>

      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Import Result</h2>
          
          {result.error ? (
            <div className="bg-red-50 border border-red-200 rounded p-4">
              <p className="text-red-800">{result.error}</p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-50 p-4 rounded">
                  <p className="text-sm text-gray-500">Total Rows</p>
                  <p className="text-2xl font-semibold">{result.total_rows}</p>
                </div>
                <div className="bg-green-50 p-4 rounded">
                  <p className="text-sm text-gray-500">Valid Rows</p>
                  <p className="text-2xl font-semibold text-green-600">{result.valid_rows}</p>
                </div>
                <div className="bg-red-50 p-4 rounded">
                  <p className="text-sm text-gray-500">Invalid Rows</p>
                  <p className="text-2xl font-semibold text-red-600">{result.invalid_rows}</p>
                </div>
              </div>

              {result.errors && result.errors.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Errors</h3>
                  <div className="bg-red-50 border border-red-200 rounded p-4 max-h-60 overflow-y-auto">
                    {result.errors.map((error: any, idx: number) => (
                      <p key={idx} className="text-sm text-red-800">
                        Row {error.row}: {error.message}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              {result.preview_rows && result.preview_rows.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Preview</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Numéros</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Bonus</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {result.preview_rows.map((row: any, idx: number) => (
                          <tr key={idx}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500">{row.draw_number || '-'}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.draw_date}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row.numbers.join(', ')}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {row.bonus_numbers && row.bonus_numbers.length > 0 ? row.bonus_numbers.join(', ') : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
