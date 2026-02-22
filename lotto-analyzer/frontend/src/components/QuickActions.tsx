import { Link } from 'react-router-dom'
import { Upload, BarChart3, Plus, Calendar } from 'lucide-react'

export default function QuickActions() {
  const actions = [
    {
      title: 'Import Data',
      description: 'Upload CSV files with draw results',
      icon: Upload,
      link: '/import',
      color: 'blue'
    },
    {
      title: 'View Draws',
      description: 'Browse historical draw data',
      icon: Calendar,
      link: '/draws',
      color: 'green'
    },
    {
      title: 'Run Analysis',
      description: 'Analyze patterns and statistics',
      icon: BarChart3,
      link: '/analysis',
      color: 'purple'
    },
    {
      title: 'Create Game',
      description: 'Configure a new lottery game',
      icon: Plus,
      link: '/games',
      color: 'orange'
    }
  ]

  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 hover:bg-blue-100',
    green: 'bg-green-50 text-green-600 hover:bg-green-100',
    purple: 'bg-purple-50 text-purple-600 hover:bg-purple-100',
    orange: 'bg-orange-50 text-orange-600 hover:bg-orange-100'
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {actions.map((action, idx) => {
          const Icon = action.icon
          return (
            <Link
              key={idx}
              to={action.link}
              className={`${colorClasses[action.color as keyof typeof colorClasses]} rounded-lg p-4 transition-colors`}
            >
              <div className="flex items-start space-x-3">
                <Icon className="w-6 h-6 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-sm">{action.title}</h3>
                  <p className="text-xs opacity-75 mt-1">{action.description}</p>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
