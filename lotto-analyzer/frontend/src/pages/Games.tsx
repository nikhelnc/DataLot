import { useState, useEffect } from 'react'
import { Plus, Settings, Edit2, Trash2, Trophy, X } from 'lucide-react'
import { gamesApi, Game, PrizeDivision } from '../api/client'

export default function Games() {
  const [games, setGames] = useState<Game[]>([])
  const [loading, setLoading] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [showDivisionsModal, setShowDivisionsModal] = useState(false)
  const [selectedGame, setSelectedGame] = useState<Game | null>(null)
  const [divisions, setDivisions] = useState<PrizeDivision[]>([])
  const [savingDivisions, setSavingDivisions] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    mainPick: 7,       // Numbers player picks
    mainDrawn: 7,      // Numbers drawn (winning numbers)
    mainMin: 1,
    mainMax: 47,
    bonusEnabled: true,
    bonusPick: 0,      // Bonus numbers player picks (0 for Oz Lotto)
    bonusDrawn: 3,     // Bonus numbers drawn (supplementary)
    bonusSeparatePool: false,
    bonusMin: 1,
    bonusMax: 20,
    frequency: 'weekly',
    days: ['TUE']
  })

  useEffect(() => {
    loadGames()
  }, [])

  const loadGames = async () => {
    try {
      const response = await gamesApi.list()
      setGames(response.data)
    } catch (error) {
      console.error('Failed to load games:', error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      const payload = {
        name: formData.name,
        description: formData.description,
        rules_json: {
          main: {
            pick: formData.mainPick,
            drawn: formData.mainDrawn,
            min: formData.mainMin,
            max: formData.mainMax
          },
          bonus: {
            enabled: formData.bonusEnabled,
            pick: formData.bonusPick,
            drawn: formData.bonusDrawn,
            separate_pool: formData.bonusSeparatePool,
            min: formData.bonusSeparatePool ? formData.bonusMin : formData.mainMin,
            max: formData.bonusSeparatePool ? formData.bonusMax : formData.mainMax
          },
          calendar: {
            expected_frequency: formData.frequency,
            days: formData.days
          }
        }
      }

      await gamesApi.create(payload)
      await loadGames()
      setShowForm(false)
      setFormData({
        name: '',
        description: '',
        mainPick: 7,
        mainDrawn: 7,
        mainMin: 1,
        mainMax: 47,
        bonusEnabled: true,
        bonusPick: 0,
        bonusDrawn: 3,
        bonusSeparatePool: false,
        bonusMin: 1,
        bonusMax: 20,
        frequency: 'weekly',
        days: ['TUE']
      })
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Failed to create game')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (gameId: string, gameName: string) => {
    if (!confirm(`Êtes-vous sûr de vouloir supprimer le jeu "${gameName}" ?\nCette action supprimera également tous les tirages et analyses associés.`)) {
      return
    }
    
    try {
      await gamesApi.delete(gameId)
      await loadGames()
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Échec de la suppression du jeu')
    }
  }

  const toggleDay = (day: string) => {
    setFormData(prev => ({
      ...prev,
      days: prev.days.includes(day)
        ? prev.days.filter(d => d !== day)
        : [...prev.days, day]
    }))
  }

  const openDivisionsModal = (game: Game) => {
    setSelectedGame(game)
    const existingDivisions = game.rules_json?.prize_divisions || []
    if (existingDivisions.length > 0) {
      setDivisions(existingDivisions)
    } else {
      // Default divisions for a typical lotto game
      setDivisions([
        { division: 1, main_numbers: 6, supplementary: 0, description: '6 Main numbers' },
        { division: 2, main_numbers: 5, supplementary: 1, description: '5 Main numbers, 1 Supplementary' },
        { division: 3, main_numbers: 5, supplementary: 0, description: '5 Main numbers' },
        { division: 4, main_numbers: 4, supplementary: 0, description: '4 Main numbers' },
        { division: 5, main_numbers: 3, supplementary: 1, description: '3 Main numbers, 1 Supplementary' },
        { division: 6, main_numbers: 3, supplementary: 0, description: '3 Main numbers' },
      ])
    }
    setShowDivisionsModal(true)
  }

  const addDivision = () => {
    const nextDiv = divisions.length + 1
    setDivisions([...divisions, { division: nextDiv, main_numbers: 0, supplementary: 0, description: '' }])
  }

  const removeDivision = (index: number) => {
    const newDivisions = divisions.filter((_, i) => i !== index)
    // Renumber divisions
    setDivisions(newDivisions.map((d, i) => ({ ...d, division: i + 1 })))
  }

  const updateDivision = (index: number, field: keyof PrizeDivision, value: number | string) => {
    const newDivisions = [...divisions]
    newDivisions[index] = { ...newDivisions[index], [field]: value }
    setDivisions(newDivisions)
  }

  const saveDivisions = async () => {
    if (!selectedGame) return
    setSavingDivisions(true)
    try {
      await gamesApi.updatePrizeDivisions(selectedGame.id, divisions)
      await loadGames()
      setShowDivisionsModal(false)
      setSelectedGame(null)
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Failed to save prize divisions')
    } finally {
      setSavingDivisions(false)
    }
  }

  const weekDays = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

  // Calculate factorial
  const factorial = (n: number): number => {
    if (n <= 1) return 1
    let result = 1
    for (let i = 2; i <= n; i++) result *= i
    return result
  }

  // Calculate combinations C(n, k) = n! / (k! * (n-k)!)
  const combinations = (n: number, k: number): number => {
    if (k > n) return 0
    if (k === 0 || k === n) return 1
    return factorial(n) / (factorial(k) * factorial(n - k))
  }

  // Calculate jackpot odds for a game
  const calculateJackpotOdds = (game: Game): string => {
    const rules = game.rules_json
    const mainPool = rules.main.max - rules.main.min + 1
    const mainPick = rules.main.pick
    
    // Main numbers combinations
    let odds = combinations(mainPool, mainPick)
    
    // If bonus is enabled and player must pick bonus numbers from separate pool
    if (rules.bonus?.enabled && rules.bonus?.separate_pool && rules.bonus?.pick > 0) {
      const bonusPool = rules.bonus.max - rules.bonus.min + 1
      const bonusPick = rules.bonus.pick
      odds *= combinations(bonusPool, bonusPick)
    }
    
    // Format the odds nicely
    if (odds >= 1_000_000_000) {
      return `1 sur ${(odds / 1_000_000_000).toFixed(1)} milliards`
    } else if (odds >= 1_000_000) {
      return `1 sur ${(odds / 1_000_000).toFixed(1)} millions`
    } else if (odds >= 1_000) {
      return `1 sur ${(odds / 1_000).toFixed(1)} milliers`
    }
    return `1 sur ${Math.round(odds)}`
  }

  return (
    <div className="px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Jeux</h1>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          <Plus className="w-5 h-5 mr-2" />
          Nouveau jeu
        </button>
      </div>

      {showForm && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Créer un nouveau jeu</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Nom du jeu
                </label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="ex: Oz_Lotto"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  required
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="ex: Lotto australien 7/47"
                />
              </div>
            </div>

            <div className="border-t pt-4">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Numéros principaux</h3>
              <div className="grid grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Choix joueur</label>
                  <input
                    type="number"
                    required
                    min="1"
                    value={formData.mainPick}
                    onChange={(e) => setFormData({ ...formData, mainPick: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tirés (gagnants)</label>
                  <input
                    type="number"
                    required
                    min="1"
                    value={formData.mainDrawn}
                    onChange={(e) => setFormData({ ...formData, mainDrawn: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Min</label>
                  <input
                    type="number"
                    required
                    min="0"
                    value={formData.mainMin}
                    onChange={(e) => setFormData({ ...formData, mainMin: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max</label>
                  <input
                    type="number"
                    required
                    min="1"
                    value={formData.mainMax}
                    onChange={(e) => setFormData({ ...formData, mainMax: parseInt(e.target.value) })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  />
                </div>
              </div>
            </div>

            <div className="border-t pt-4">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Numéros bonus</h3>
              <div className="mb-3">
                <label className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={formData.bonusEnabled}
                    onChange={(e) => setFormData({ ...formData, bonusEnabled: e.target.checked })}
                    className="form-checkbox"
                  />
                  <span className="ml-2">Activer les bonus</span>
                </label>
              </div>
              {formData.bonusEnabled && (
                <>
                  <div className="mb-3">
                    <label className="inline-flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.bonusSeparatePool}
                        onChange={(e) => setFormData({ ...formData, bonusSeparatePool: e.target.checked })}
                        className="form-checkbox"
                      />
                      <span className="ml-2">Pool séparé (ex: Powerball)</span>
                    </label>
                  </div>
                  <div className="grid grid-cols-4 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Choix joueur</label>
                      <input
                        type="number"
                        required
                        min="0"
                        value={formData.bonusPick}
                        onChange={(e) => setFormData({ ...formData, bonusPick: parseInt(e.target.value) })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      />
                      <p className="text-xs text-gray-500 mt-1">0 pour Oz Lotto</p>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Tirés (complémentaires)</label>
                      <input
                        type="number"
                        required
                        min="0"
                        value={formData.bonusDrawn}
                        onChange={(e) => setFormData({ ...formData, bonusDrawn: parseInt(e.target.value) })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      />
                    </div>
                    {formData.bonusSeparatePool && (
                      <>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Min</label>
                          <input
                            type="number"
                            required
                            min="0"
                            value={formData.bonusMin}
                            onChange={(e) => setFormData({ ...formData, bonusMin: parseInt(e.target.value) })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Max</label>
                          <input
                            type="number"
                            required
                            min="1"
                            value={formData.bonusMax}
                            onChange={(e) => setFormData({ ...formData, bonusMax: parseInt(e.target.value) })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md"
                          />
                        </div>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>

            <div className="border-t pt-4">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Calendrier</h3>
              <div className="mb-3">
                <label className="block text-sm font-medium text-gray-700 mb-1">Fréquence</label>
                <select
                  value={formData.frequency}
                  onChange={(e) => setFormData({ ...formData, frequency: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="daily">Quotidien</option>
                  <option value="weekly">Hebdomadaire</option>
                  <option value="biweekly">Bi-hebdomadaire</option>
                  <option value="triweekly">Tri-hebdomadaire</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Jours de tirage</label>
                <div className="flex flex-wrap gap-2">
                  {weekDays.map(day => (
                    <button
                      key={day}
                      type="button"
                      onClick={() => toggleDay(day)}
                      className={`px-3 py-1 rounded-md text-sm font-medium ${
                        formData.days.includes(day)
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      {day}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-3 pt-4">
              <button
                type="button"
                onClick={() => setShowForm(false)}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
              >
                Annuler
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400"
              >
                {loading ? 'Création...' : 'Créer le jeu'}
              </button>
            </div>
          </form>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {games.map(game => (
          <div key={game.id} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{game.name}</h3>
                <p className="text-sm text-gray-500">{game.description}</p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={() => openDivisionsModal(game)}
                  className="p-2 text-amber-600 hover:bg-amber-50 rounded-md transition-colors"
                  title="Configurer les divisions de gains"
                >
                  <Trophy className="w-4 h-4" />
                </button>
                <button
                  onClick={() => alert('La modification des jeux sera disponible dans une prochaine version.\n\nPour modifier un jeu, vous devez actuellement :\n1. Créer un nouveau jeu avec les paramètres souhaités\n2. Réimporter vos données pour le nouveau jeu\n\nNote: Les jeux existants ne peuvent pas être modifiés pour préserver l\'intégrité des données historiques.')}
                  className="p-2 text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                  title="Modifier le jeu"
                >
                  <Edit2 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleDelete(game.id, game.name)}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-md transition-colors"
                  title="Supprimer le jeu"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Principaux:</span>
                <span className="font-medium">
                  Choix {game.rules_json.main.pick}, Tirés {game.rules_json.main.drawn || game.rules_json.main.pick} de {game.rules_json.main.min}-{game.rules_json.main.max}
                </span>
              </div>
              {game.rules_json.bonus?.enabled && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Bonus:</span>
                  <span className="font-medium">
                    Choix {game.rules_json.bonus.pick || 0}, Tirés {game.rules_json.bonus.drawn || game.rules_json.bonus.pick || 0}
                    {game.rules_json.bonus.separate_pool 
                      ? ` de ${game.rules_json.bonus.min}-${game.rules_json.bonus.max}` 
                      : ' (même pool)'}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-600">Fréquence:</span>
                <span className="font-medium capitalize">{game.rules_json.calendar.expected_frequency}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Jours:</span>
                <span className="font-medium">{game.rules_json.calendar.days.join(', ')}</span>
              </div>
              <div className="flex justify-between pt-2 border-t mt-2">
                <span className="text-gray-600">🎰 Jackpot:</span>
                <span className="font-medium text-purple-600">{calculateJackpotOdds(game)}</span>
              </div>
              <div className="pt-2 border-t">
                <span className="text-xs text-gray-400">Version {game.version}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {games.length === 0 && !showForm && (
        <div className="text-center py-12">
          <Settings className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">Aucun jeu</h3>
          <p className="mt-1 text-sm text-gray-500">Commencez par créer un nouveau jeu.</p>
        </div>
      )}

      {/* Prize Divisions Modal */}
      {showDivisionsModal && selectedGame && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                <Trophy className="w-5 h-5 text-amber-500" />
                Divisions de gains - {selectedGame.name}
              </h2>
              <button
                onClick={() => { setShowDivisionsModal(false); setSelectedGame(null); }}
                className="p-2 hover:bg-gray-100 rounded-md"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm font-medium text-gray-500 border-b">
                    <th className="pb-2 w-16">Div</th>
                    <th className="pb-2">Principaux</th>
                    <th className="pb-2">Supplémentaires</th>
                    <th className="pb-2">Description</th>
                    <th className="pb-2 w-12"></th>
                  </tr>
                </thead>
                <tbody>
                  {divisions.map((div, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="py-2 font-medium text-gray-900">{div.division}</td>
                      <td className="py-2">
                        <input
                          type="number"
                          min="0"
                          value={div.main_numbers}
                          onChange={(e) => updateDivision(idx, 'main_numbers', parseInt(e.target.value) || 0)}
                          className="w-16 px-2 py-1 border rounded text-center"
                        />
                      </td>
                      <td className="py-2">
                        <input
                          type="number"
                          min="0"
                          value={div.supplementary}
                          onChange={(e) => updateDivision(idx, 'supplementary', parseInt(e.target.value) || 0)}
                          className="w-16 px-2 py-1 border rounded text-center"
                        />
                      </td>
                      <td className="py-2">
                        <input
                          type="text"
                          value={div.description || ''}
                          onChange={(e) => updateDivision(idx, 'description', e.target.value)}
                          className="w-full px-2 py-1 border rounded"
                          placeholder="ex: 6 Main numbers"
                        />
                      </td>
                      <td className="py-2">
                        <button
                          onClick={() => removeDivision(idx)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                          title="Supprimer"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              <button
                onClick={addDivision}
                className="mt-4 flex items-center gap-2 px-3 py-2 text-blue-600 hover:bg-blue-50 rounded-md"
              >
                <Plus className="w-4 h-4" />
                Ajouter une division
              </button>
            </div>
            
            <div className="flex justify-end gap-3 p-4 border-t bg-gray-50">
              <button
                onClick={() => { setShowDivisionsModal(false); setSelectedGame(null); }}
                className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100"
              >
                Annuler
              </button>
              <button
                onClick={saveDivisions}
                disabled={savingDivisions}
                className="px-4 py-2 bg-amber-600 text-white rounded-md hover:bg-amber-700 disabled:bg-gray-400"
              >
                {savingDivisions ? 'Enregistrement...' : 'Enregistrer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
