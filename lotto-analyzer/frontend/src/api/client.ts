import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface Game {
  id: string
  name: string
  description?: string
  rules_json: any
  version: number
  created_at: string
}

export interface Draw {
  id: string
  game_id: string
  draw_number?: number
  draw_date: string
  numbers: number[]
  bonus_numbers?: number[]
  created_at: string
}

export interface Analysis {
  analysis_id: string
  game_id: string
  name: string
  dataset_hash?: string
  code_version?: string
  params?: any
  results?: any
  created_at: string
}

export interface Alert {
  id: string
  game_id: string
  analysis_id?: string
  severity: 'low' | 'medium' | 'high'
  score: number
  message: string
  evidence_json?: any
  created_at: string
}

export interface PrizeDivision {
  division: number
  main_numbers: number
  supplementary: number
  description?: string
}

export const gamesApi = {
  list: () => apiClient.get<Game[]>('/games'),
  get: (id: string) => apiClient.get<Game>(`/games/${id}`),
  create: (data: any) => apiClient.post<Game>('/games', data),
  update: (id: string, data: any) => apiClient.put<Game>(`/games/${id}`, data),
  updatePrizeDivisions: (id: string, prizeDivisions: PrizeDivision[]) => 
    apiClient.put<Game>(`/games/${id}/prize-divisions`, { prize_divisions: prizeDivisions }),
  delete: (id: string) => apiClient.delete(`/games/${id}`),
}

export interface DrawCreate {
  game_id: string
  draw_number?: number
  draw_date: string
  numbers: number[]
  bonus_numbers?: number[]
}

export const drawsApi = {
  list: (gameIdOrParams?: string | any, params?: any) => {
    if (typeof gameIdOrParams === 'string') {
      return apiClient.get<Draw[]>('/draws', { params: { game_id: gameIdOrParams, ...params } })
    }
    return apiClient.get<Draw[]>('/draws', { params: gameIdOrParams })
  },
  get: (id: string) => apiClient.get<Draw>(`/draws/${id}`),
  create: (data: DrawCreate) => apiClient.post<Draw>('/draws', data),
  delete: (id: string) => apiClient.delete(`/draws/${id}`),
}

export const importsApi = {
  upload: (gameId: string, file: File, mode: 'preview' | 'commit') => {
    const formData = new FormData()
    formData.append('file', file)
    return apiClient.post(`/draws/import?game_id=${gameId}&mode=${mode}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  getStatus: (id: string) => apiClient.get(`/draws/import/${id}`),
}

export interface NextPrediction {
  main_numbers: number[]
  bonus_numbers: number[]
  combinations: { main_numbers: number[], bonus_numbers: number[] }[]
  excluded_main: number[]
  excluded_bonus: number[]
  models_used: string[]
  total_draws_analyzed: number
}

export const analysesApi = {
  run: (data: any) => apiClient.post<Analysis>('/analyses/run', data),
  get: (id: string) => apiClient.get<Analysis>(`/analyses/${id}`),
  exportCsv: (id: string) => apiClient.get(`/analyses/${id}/export.csv`, { responseType: 'blob' }),
  getReport: (id: string) => apiClient.get(`/analyses/${id}/report.html`, { responseType: 'text' }),
  getNextPrediction: (gameId: string, nCombinations: number = 10) => 
    apiClient.get<NextPrediction>(`/analyses/predict/${gameId}?n_combinations=${nCombinations}`),
}

export const alertsApi = {
  list: (params?: any) => apiClient.get<Alert[]>('/alerts', { params }),
  get: (id: string) => apiClient.get<Alert>(`/alerts/${id}`),
}

export const healthApi = {
  check: () => apiClient.get('/health'),
}

// Alias for compatibility with new pages
export const api = apiClient
