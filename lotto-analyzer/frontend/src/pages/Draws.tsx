import { useState, useEffect } from 'react'
import { Calendar, Plus, X, Trash2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { gamesApi, drawsApi, Game, DrawCreate } from '../api/client'

interface Draw {
  id: string
  game_id: string
  draw_number: number | null
  draw_date: string
  numbers: number[]
  bonus_numbers: number[]
  created_at: string
}

export default function Draws() {
  const { t } = useTranslation()
  const [games, setGames] = useState<Game[]>([])
  const [draws, setDraws] = useState<Draw[]>([])
  const [selectedGameId, setSelectedGameId] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    limit: 50,
    offset: 0
  })
  const [total, setTotal] = useState(0)
  
  // Modal state for adding a draw
  const [showAddModal, setShowAddModal] = useState(false)
  const [addDrawForm, setAddDrawForm] = useState<{
    draw_number: string
    draw_date: string
    numbers: string[]
    bonus_numbers: string[]
  }>({
    draw_number: '',
    draw_date: new Date().toISOString().split('T')[0],
    numbers: [],
    bonus_numbers: []
  })
  const [addError, setAddError] = useState('')
  const [addLoading, setAddLoading] = useState(false)

  useEffect(() => {
    loadGames()
  }, [])

  useEffect(() => {
    if (selectedGameId) {
      loadDraws()
    }
  }, [selectedGameId, filters])

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
      const params: any = {
        game_id: selectedGameId,
        page_size: filters.limit,
        page: Math.floor(filters.offset / filters.limit) + 1
      }
      if (filters.startDate) params.from_date = filters.startDate
      if (filters.endDate) params.to_date = filters.endDate

      const response = await drawsApi.list(params)
      setDraws(response.data)
      setTotal(response.data.length)
    } catch (error) {
      console.error('Failed to load draws:', error)
    } finally {
      setLoading(false)
    }
  }

  const handlePrevPage = () => {
    setFilters(prev => ({
      ...prev,
      offset: Math.max(0, prev.offset - prev.limit)
    }))
  }

  const handleNextPage = () => {
    setFilters(prev => ({
      ...prev,
      offset: prev.offset + prev.limit
    }))
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  }

  const currentGame = games.find(g => g.id === selectedGameId)

  // Initialize form when game changes or modal opens
  const initializeAddForm = () => {
    if (!currentGame) return
    
    const mainCount = currentGame.rules_json?.main_numbers?.count || 
                      currentGame.rules_json?.main?.pick || 
                      currentGame.rules_json?.main?.drawn || 6
    
    // Check if bonus is enabled and get the count
    const bonusEnabled = currentGame.rules_json?.bonus?.enabled || 
                         currentGame.rules_json?.bonus_numbers?.count > 0
    const bonusCount = bonusEnabled ? (
                       currentGame.rules_json?.bonus_numbers?.count || 
                       currentGame.rules_json?.bonus?.drawn ||
                       currentGame.rules_json?.bonus?.pick || 0) : 0
    
    console.log('Game rules:', currentGame.rules_json)
    console.log('Main count:', mainCount, 'Bonus count:', bonusCount)
    
    setAddDrawForm({
      draw_number: '',
      draw_date: new Date().toISOString().split('T')[0],
      numbers: Array(mainCount).fill(''),
      bonus_numbers: Array(bonusCount).fill('')
    })
    setAddError('')
  }

  const openAddModal = () => {
    initializeAddForm()
    setShowAddModal(true)
  }

  const handleAddDraw = async () => {
    if (!currentGame || !selectedGameId) return
    
    setAddError('')
    setAddLoading(true)
    
    try {
      // Parse and validate numbers
      const numbers = addDrawForm.numbers.map(n => parseInt(n.trim())).filter(n => !isNaN(n))
      const bonusNumbers = addDrawForm.bonus_numbers.map(n => parseInt(n.trim())).filter(n => !isNaN(n))
      
      const mainCount = currentGame.rules_json?.main_numbers?.count || 
                        currentGame.rules_json?.main?.pick || 
                        currentGame.rules_json?.main?.drawn || 6
      const bonusEnabled = currentGame.rules_json?.bonus?.enabled || 
                           currentGame.rules_json?.bonus_numbers?.count > 0
      const bonusCount = bonusEnabled ? (
                         currentGame.rules_json?.bonus_numbers?.count || 
                         currentGame.rules_json?.bonus?.drawn ||
                         currentGame.rules_json?.bonus?.pick || 0) : 0
      
      if (numbers.length !== mainCount) {
        setAddError(`Veuillez saisir ${mainCount} numéros principaux`)
        setAddLoading(false)
        return
      }
      
      if (bonusCount > 0 && bonusNumbers.length !== bonusCount) {
        setAddError(`Veuillez saisir ${bonusCount} numéros bonus`)
        setAddLoading(false)
        return
      }
      
      if (!addDrawForm.draw_date) {
        setAddError('Veuillez saisir une date')
        setAddLoading(false)
        return
      }
      
      const drawData: DrawCreate = {
        game_id: selectedGameId,
        draw_number: addDrawForm.draw_number ? parseInt(addDrawForm.draw_number) : undefined,
        draw_date: new Date(addDrawForm.draw_date).toISOString(),
        numbers,
        bonus_numbers: bonusNumbers.length > 0 ? bonusNumbers : undefined
      }
      
      await drawsApi.create(drawData)
      setShowAddModal(false)
      loadDraws() // Refresh the list
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Erreur lors de la création du tirage'
      setAddError(message)
    } finally {
      setAddLoading(false)
    }
  }

  const updateNumber = (index: number, value: string, isBonus: boolean = false) => {
    if (isBonus) {
      const newBonusNumbers = [...addDrawForm.bonus_numbers]
      newBonusNumbers[index] = value
      setAddDrawForm({ ...addDrawForm, bonus_numbers: newBonusNumbers })
    } else {
      const newNumbers = [...addDrawForm.numbers]
      newNumbers[index] = value
      setAddDrawForm({ ...addDrawForm, numbers: newNumbers })
    }
  }

  const handleDeleteDraw = async (drawId: string) => {
    if (!confirm('Êtes-vous sûr de vouloir supprimer ce tirage ?')) {
      return
    }
    
    try {
      await drawsApi.delete(drawId)
      loadDraws() // Refresh the list
    } catch (error: any) {
      console.error('Failed to delete draw:', error)
      alert('Erreur lors de la suppression du tirage')
    }
  }

  return (
    <div className="px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">{t('draws.title')}</h1>
        {selectedGameId && (
          <button
            onClick={openAddModal}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
          >
            <Plus className="h-5 w-5 mr-2" />
            Ajouter un tirage
          </button>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('draws.selectGame')}
            </label>
            <select
              value={selectedGameId}
              onChange={(e) => setSelectedGameId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
              {t('draws.startDate')}
            </label>
            <input
              type="date"
              value={filters.startDate}
              onChange={(e) => setFilters({ ...filters, startDate: e.target.value, offset: 0 })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('draws.endDate')}
            </label>
            <input
              type="date"
              value={filters.endDate}
              onChange={(e) => setFilters({ ...filters, endDate: e.target.value, offset: 0 })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {t('common.filter')}
            </label>
            <select
              value={filters.limit}
              onChange={(e) => setFilters({ ...filters, limit: parseInt(e.target.value), offset: 0 })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">{t('draws.loading')}</p>
        </div>
      ) : draws.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow">
          <Calendar className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">{t('draws.noDraws')}</h3>
          <p className="mt-1 text-sm text-gray-500">
            {selectedGameId ? t('import.uploadFile') : t('draws.selectGame')}
          </p>
        </div>
      ) : (
        <>
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
              <div className="flex justify-between items-center">
                <h2 className="text-lg font-semibold text-gray-900">
                  {currentGame?.name} - {total} draws
                </h2>
                <div className="text-sm text-gray-500">
                  Page {Math.floor(filters.offset / filters.limit) + 1}
                </div>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      #
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {t('draws.date')}
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {t('draws.numbers')}
                    </th>
                    {currentGame?.rules_json.bonus?.enabled && (
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {t('draws.bonus')}
                      </th>
                    )}
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      {t('draws.sum')}
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {draws.map((draw) => {
                    const sum = draw.numbers.reduce((a, b) => a + b, 0)
                    return (
                      <tr key={draw.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-500">
                          {draw.draw_number || '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatDate(draw.draw_date)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex space-x-2">
                            {draw.numbers.map((num, idx) => (
                              <span
                                key={idx}
                                className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-800 text-sm font-semibold"
                              >
                                {num}
                              </span>
                            ))}
                          </div>
                        </td>
                        {currentGame?.rules_json.bonus?.enabled && (
                          <td className="px-6 py-4 whitespace-nowrap">
                            {draw.bonus_numbers && draw.bonus_numbers.length > 0 ? (
                              <div className="flex space-x-2">
                                {draw.bonus_numbers.map((bonus, idx) => (
                                  <span key={idx} className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-yellow-100 text-yellow-800 text-sm font-semibold">
                                    {bonus}
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className="text-gray-400">-</span>
                            )}
                          </td>
                        )}
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {sum}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right">
                          <button
                            onClick={() => handleDeleteDraw(draw.id)}
                            className="text-red-600 hover:text-red-800 p-1 rounded hover:bg-red-50"
                            title="Supprimer ce tirage"
                          >
                            <Trash2 className="h-5 w-5" />
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
              <div className="flex justify-between items-center">
                <button
                  onClick={handlePrevPage}
                  disabled={filters.offset === 0}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {t('common.reset')}
                </button>
                <span className="text-sm text-gray-700">
                  Showing {filters.offset + 1} - {Math.min(filters.offset + filters.limit, filters.offset + total)}
                </span>
                <button
                  onClick={handleNextPage}
                  disabled={total < filters.limit}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {t('common.apply')}
                </button>
              </div>
            </div>
          </div>

          {currentGame && (
            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">Game Configuration</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-blue-800">
                <div>
                  <span className="font-medium">Main:</span> Pick {currentGame.rules_json.main?.pick || '?'}, Draw {currentGame.rules_json.main?.drawn || currentGame.rules_json.main?.pick || '?'} from {currentGame.rules_json.main?.min || 1}-{currentGame.rules_json.main?.max || 49}
                </div>
                {currentGame.rules_json.bonus?.enabled && (
                  <div>
                    <span className="font-medium">Bonus:</span> Pick {currentGame.rules_json.bonus.pick || 0}, Draw {currentGame.rules_json.bonus.drawn || currentGame.rules_json.bonus.pick || 0}
                    {currentGame.rules_json.bonus.separate_pool ? ` from ${currentGame.rules_json.bonus.min}-${currentGame.rules_json.bonus.max}` : ' (same pool)'}
                  </div>
                )}
                <div>
                  <span className="font-medium">Frequency:</span> {currentGame.rules_json.calendar?.expected_frequency || 'weekly'}
                </div>
                <div>
                  <span className="font-medium">Days:</span> {currentGame.rules_json.calendar?.days?.join(', ') || '-'}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Modal d'ajout de tirage */}
      {showAddModal && currentGame && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg mx-4">
            <div className="flex justify-between items-center px-6 py-4 border-b">
              <h3 className="text-lg font-semibold text-gray-900">
                Ajouter un tirage - {currentGame.name}
              </h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            <div className="px-6 py-4 space-y-4">
              {addError && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md text-sm">
                  {addError}
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Numéro du tirage
                  </label>
                  <input
                    type="number"
                    value={addDrawForm.draw_number}
                    onChange={(e) => setAddDrawForm({ ...addDrawForm, draw_number: e.target.value })}
                    placeholder="Ex: 1234"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Date du tirage *
                  </label>
                  <input
                    type="date"
                    value={addDrawForm.draw_date}
                    onChange={(e) => setAddDrawForm({ ...addDrawForm, draw_date: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Numéros principaux * ({addDrawForm.numbers.length} numéros)
                </label>
                <div className="flex flex-wrap gap-2">
                  {addDrawForm.numbers.map((num, idx) => (
                    <input
                      key={idx}
                      type="number"
                      value={num}
                      onChange={(e) => updateNumber(idx, e.target.value, false)}
                      placeholder={`N${idx + 1}`}
                      className="w-16 px-2 py-2 border border-gray-300 rounded-md text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                      min={currentGame.rules_json?.main_numbers?.min || currentGame.rules_json?.main?.min || 1}
                      max={currentGame.rules_json?.main_numbers?.max || currentGame.rules_json?.main?.max || 49}
                    />
                  ))}
                </div>
              </div>

              {addDrawForm.bonus_numbers.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Numéros bonus * ({addDrawForm.bonus_numbers.length} numéros)
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {addDrawForm.bonus_numbers.map((num, idx) => (
                      <input
                        key={idx}
                        type="number"
                        value={num}
                        onChange={(e) => updateNumber(idx, e.target.value, true)}
                        placeholder={`B${idx + 1}`}
                        className="w-16 px-2 py-2 border border-yellow-300 rounded-md text-center focus:outline-none focus:ring-2 focus:ring-yellow-500 bg-yellow-50"
                        min={currentGame.rules_json?.bonus_numbers?.min || currentGame.rules_json?.bonus?.min || 1}
                        max={currentGame.rules_json?.bonus_numbers?.max || currentGame.rules_json?.bonus?.max || 20}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 px-6 py-4 border-t bg-gray-50">
              <button
                onClick={() => setShowAddModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100"
              >
                Annuler
              </button>
              <button
                onClick={handleAddDraw}
                disabled={addLoading}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {addLoading ? 'Enregistrement...' : 'Enregistrer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
