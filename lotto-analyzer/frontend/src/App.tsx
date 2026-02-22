import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, Calendar, Upload, BarChart3, AlertTriangle, TrendingUp, Settings, Layers, Target, Shield, DollarSign, GitCompare, Zap, ChevronLeft, ChevronRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import Games from './pages/Games'
import Draws from './pages/Draws'
import Import from './pages/Import'
import Analysis from './pages/Analysis'
import Alerts from './pages/Alerts'
import Probabilities from './pages/Probabilities'
import AdvancedModels from './pages/AdvancedModels'
import Backtest from './pages/Backtest'
import Forensics from './pages/Forensics'
import Fraud from './pages/Fraud'
import Jackpot from './pages/Jackpot'
import LanguageSwitcher from './components/LanguageSwitcher'
import './i18n/config'

interface NavItemProps {
  to: string
  icon: React.ReactNode
  label: string
  collapsed: boolean
}

function NavItem({ to, icon, label, collapsed }: NavItemProps) {
  const location = useLocation()
  const isActive = location.pathname === to
  
  return (
    <Link
      to={to}
      className={`flex items-center px-3 py-2.5 rounded-lg transition-all duration-200 group ${
        isActive 
          ? 'bg-blue-600 text-white shadow-md' 
          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
      }`}
      title={collapsed ? label : undefined}
    >
      <span className={`flex-shrink-0 ${isActive ? 'text-white' : 'text-gray-500 group-hover:text-gray-700'}`}>
        {icon}
      </span>
      {!collapsed && (
        <span className="ml-3 text-sm font-medium truncate">{label}</span>
      )}
    </Link>
  )
}

interface NavSectionProps {
  title: string
  children: React.ReactNode
  collapsed: boolean
}

function NavSection({ title, children, collapsed }: NavSectionProps) {
  return (
    <div className="mb-6">
      {!collapsed && (
        <h3 className="px-3 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          {title}
        </h3>
      )}
      <nav className="space-y-1">
        {children}
      </nav>
    </div>
  )
}

function AppContent() {
  const { t } = useTranslation()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside className={`${sidebarCollapsed ? 'w-16' : 'w-64'} bg-white border-r border-gray-200 flex flex-col transition-all duration-300 fixed h-full z-30`}>
        {/* Logo */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200">
          {!sidebarCollapsed && (
            <div className="flex items-center">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Target className="w-5 h-5 text-white" />
              </div>
              <span className="ml-3 text-lg font-bold text-gray-900">Lotto Analyzer</span>
            </div>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700"
          >
            {sidebarCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto py-4 px-2">
          <NavSection title={t('nav.sections.general', 'Général')} collapsed={sidebarCollapsed}>
            <NavItem to="/" icon={<Home className="w-5 h-5" />} label={t('nav.dashboard')} collapsed={sidebarCollapsed} />
            <NavItem to="/games" icon={<Settings className="w-5 h-5" />} label={t('nav.games')} collapsed={sidebarCollapsed} />
            <NavItem to="/draws" icon={<Calendar className="w-5 h-5" />} label={t('nav.draws')} collapsed={sidebarCollapsed} />
            <NavItem to="/import" icon={<Upload className="w-5 h-5" />} label={t('nav.import')} collapsed={sidebarCollapsed} />
          </NavSection>

          <NavSection title={t('nav.sections.analysis', 'Analyse')} collapsed={sidebarCollapsed}>
            <NavItem to="/analysis" icon={<BarChart3 className="w-5 h-5" />} label={t('nav.analysis')} collapsed={sidebarCollapsed} />
            <NavItem to="/alerts" icon={<AlertTriangle className="w-5 h-5" />} label={t('nav.alerts')} collapsed={sidebarCollapsed} />
            <NavItem to="/probabilities" icon={<TrendingUp className="w-5 h-5" />} label={t('nav.probabilities')} collapsed={sidebarCollapsed} />
          </NavSection>

          <NavSection title={t('nav.sections.models', 'Modèles')} collapsed={sidebarCollapsed}>
            <NavItem to="/advanced-models" icon={<Layers className="w-5 h-5" />} label={t('nav.advancedModels')} collapsed={sidebarCollapsed} />
            <NavItem to="/backtest" icon={<Target className="w-5 h-5" />} label={t('nav.backtest')} collapsed={sidebarCollapsed} />
          </NavSection>

          <NavSection title={t('nav.sections.forensics', 'Forensique')} collapsed={sidebarCollapsed}>
            <NavItem to="/forensics" icon={<Shield className="w-5 h-5" />} label={t('nav.forensics', 'Forensique')} collapsed={sidebarCollapsed} />
            <NavItem to="/fraud" icon={<AlertTriangle className="w-5 h-5" />} label={t('nav.fraud', 'Détection Fraude')} collapsed={sidebarCollapsed} />
            <NavItem to="/jackpot" icon={<DollarSign className="w-5 h-5" />} label={t('nav.jackpot', 'Analyse Jackpot')} collapsed={sidebarCollapsed} />
          </NavSection>

          <NavSection title={t('nav.sections.tools', 'Outils')} collapsed={sidebarCollapsed}>
            <NavItem to="/compare" icon={<GitCompare className="w-5 h-5" />} label={t('nav.compare', 'Comparateur')} collapsed={sidebarCollapsed} />
            <NavItem to="/power-analysis" icon={<Zap className="w-5 h-5" />} label={t('nav.powerAnalysis', 'Puissance Stat.')} collapsed={sidebarCollapsed} />
          </NavSection>
        </div>

        {/* Language Switcher & Footer */}
        <div className="border-t border-gray-200 p-3">
          <div className={`flex ${sidebarCollapsed ? 'justify-center' : 'justify-between'} items-center`}>
            {!sidebarCollapsed && (
              <span className="text-xs text-gray-400">v2.0</span>
            )}
            <LanguageSwitcher />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className={`flex-1 ${sidebarCollapsed ? 'ml-16' : 'ml-64'} transition-all duration-300`}>
        {/* Top Bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 sticky top-0 z-20">
          <div className="flex items-center">
            <h2 className="text-lg font-semibold text-gray-800">
              {/* Page title will be set by each page */}
            </h2>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-500">
              {new Date().toLocaleDateString('fr-FR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/games" element={<Games />} />
            <Route path="/draws" element={<Draws />} />
            <Route path="/import" element={<Import />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/probabilities" element={<Probabilities />} />
            <Route path="/advanced-models" element={<AdvancedModels />} />
            <Route path="/backtest" element={<Backtest />} />
            {/* New v2.0 routes - placeholder pages */}
            <Route path="/forensics" element={<Forensics />} />
            <Route path="/fraud" element={<Fraud />} />
            <Route path="/jackpot" element={<Jackpot />} />
            <Route path="/compare" element={<PlaceholderPage title="Comparateur Inter-Loteries" description="Comparaison entre différentes loteries - En cours de développement" />} />
            <Route path="/power-analysis" element={<PlaceholderPage title="Calculateur de Puissance" description="Calcul de puissance statistique - En cours de développement" />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t mt-6">
          <div className="px-6 py-6">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-yellow-900 mb-2">
                ⚠️ {t('footer.limits')}
              </h3>
              <div className="text-xs text-yellow-800 space-y-1">
                <p>{t('footer.line1')} {t('footer.line2')}</p>
                <p>{t('footer.line3')} {t('footer.line4')}</p>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}

// Placeholder component for new pages
function PlaceholderPage({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh]">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center max-w-lg">
        <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <Shield className="w-8 h-8 text-blue-600" />
        </div>
        <h1 className="text-2xl font-bold text-gray-900 mb-3">{title}</h1>
        <p className="text-gray-500 mb-6">{description}</p>
        <div className="inline-flex items-center px-4 py-2 bg-blue-50 text-blue-700 rounded-lg text-sm font-medium">
          <Zap className="w-4 h-4 mr-2" />
          Bientôt disponible
        </div>
      </div>
    </div>
  )
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App
