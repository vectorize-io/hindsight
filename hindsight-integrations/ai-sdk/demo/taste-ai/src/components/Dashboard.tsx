'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface Recipe {
  name: string;
  emoji: string;
  description?: string;
  healthScore: number;
  timeMinutes: number;
  ingredients: string[];
  instructions: string;
  tags: string[];
  mealType?: string;
  date?: string;
}

interface DashboardProps {
  username: string;
  onEatNow: () => void;
  onOpenPreferences: () => void;
  onViewRecipe: (recipe: Recipe) => void;
  refreshKey: number;
}

interface HealthData {
  score: number;
  trend: 'up' | 'down' | 'stable';
  insight: string;
}

interface Meal {
  id: string;
  name: string;
  type: string;
  date: string;
  emoji: string;
  description?: string;
  healthScore?: number;
  timeMinutes?: number;
  ingredients?: string[];
  instructions?: string;
  tags?: string[];
}

interface UserPreferences {
  language?: string;
  cuisines?: string[];
  dietary?: string[];
  goals?: string[];
  dislikes?: string[];
}

export default function Dashboard({ username, onEatNow, onOpenPreferences, onViewRecipe, refreshKey }: DashboardProps) {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [meals, setMeals] = useState<Meal[]>([]);
  const [preferences, setPreferences] = useState<UserPreferences>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, [refreshKey]);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/dashboard?username=${encodeURIComponent(username)}`);
      const data = await res.json();
      setHealth(data.health);
      setMeals(data.meals);
      setPreferences(data.preferences || {});
    } catch (e) {
      console.error('Failed to load dashboard:', e);
    }
    setLoading(false);
  };

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-500';
    if (score >= 6) return 'text-yellow-500';
    if (score >= 4) return 'text-orange-500';
    return 'text-red-500';
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '→';
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  };

  return (
    <div className="min-h-screen px-4 py-6 max-w-md mx-auto">
      {/* Header */}
      <header className="flex items-start justify-between mb-8">
        <div className="flex-1">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-500 to-rose-500 bg-clip-text text-transparent">
            Let me cook 👨‍🍳
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            <span className="font-medium text-gray-700">{username}</span>'s personal food memory
          </p>
        </div>
        <button
          onClick={onOpenPreferences}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-white shadow-sm border border-gray-100 hover:bg-gray-50 transition-colors"
          title="Preferences"
        >
          ⚙️
        </button>
      </header>

      {/* Preferences Hint */}
      {(preferences.cuisines?.length || preferences.dietary?.length || preferences.goals?.length) && (
        <div className="mb-6 p-3 bg-orange-50 border border-orange-100 rounded-lg">
          <div className="flex items-start gap-2">
            <span className="text-sm">💡</span>
            <div className="flex-1">
              <p className="text-xs text-gray-600">
                {preferences.cuisines?.length ? (
                  <span className="mr-2">
                    <strong>Cuisines:</strong> {preferences.cuisines.join(', ')}
                  </span>
                ) : null}
                {preferences.dietary?.length ? (
                  <span className="mr-2">
                    <strong>Diet:</strong> {preferences.dietary.join(', ')}
                  </span>
                ) : null}
                {preferences.goals?.length ? (
                  <span>
                    <strong>Goals:</strong> {preferences.goals.join(', ')}
                  </span>
                ) : null}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Health Score Circle */}
      <motion.div
        className="flex flex-col items-center mb-8"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <div className="relative w-40 h-40">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke="#f3f4f6"
              strokeWidth="12"
              fill="none"
            />
            {health && (
              <motion.circle
                cx="80"
                cy="80"
                r="70"
                stroke="url(#gradient)"
                strokeWidth="12"
                fill="none"
                strokeLinecap="round"
                initial={{ strokeDasharray: '0 440' }}
                animate={{
                  strokeDasharray: `${(health.score / 10) * 440} 440`
                }}
                transition={{ duration: 1, delay: 0.5 }}
              />
            )}
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" />
                <stop offset="100%" stopColor="#ec4899" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            {loading ? (
              <div className="w-8 h-8 border-4 border-orange-200 border-t-orange-500 rounded-full animate-spin" />
            ) : health ? (
              <>
                <span className={`text-4xl font-bold ${getScoreColor(health.score)}`}>
                  {health.score.toFixed(1)}
                </span>
                <span className="text-gray-400 text-sm">Health Score</span>
                <span className={`text-lg ${health.trend === 'up' ? 'text-green-500' : health.trend === 'down' ? 'text-red-500' : 'text-gray-400'}`}>
                  {getTrendIcon(health.trend)}
                </span>
              </>
            ) : (
              <>
                <span className="text-3xl">🍽️</span>
                <span className="text-gray-400 text-sm text-center px-2">Log a meal to see your score</span>
              </>
            )}
          </div>
        </div>

        {health?.insight && (
          <motion.p
            className="text-center text-gray-600 mt-4 text-sm px-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            {health.insight}
          </motion.p>
        )}
      </motion.div>

      {/* Eat Now Button */}
      <motion.button
        onClick={onEatNow}
        className="w-full py-4 bg-gradient-to-r from-orange-500 to-rose-500 text-white text-xl font-semibold rounded-2xl shadow-lg shadow-orange-200 mb-8"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        🍽️ Eat Now
      </motion.button>

      {/* Recent Meals Timeline */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-800 mb-4">Recent Meals</h2>
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-16 bg-gray-100 rounded-xl animate-pulse" />
            ))}
          </div>
        ) : meals.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <p className="text-4xl mb-2">🍽️</p>
            <p>No meals logged yet</p>
            <p className="text-sm">Tap "Eat Now" to get started!</p>
          </div>
        ) : (
          <div className="space-y-3">
            {meals.map((meal, idx) => (
              <motion.button
                key={meal.id}
                onClick={() => {
                  if (meal.ingredients && meal.instructions) {
                    onViewRecipe({
                      name: meal.name,
                      emoji: meal.emoji,
                      description: meal.description,
                      healthScore: meal.healthScore || 0,
                      timeMinutes: meal.timeMinutes || 0,
                      ingredients: meal.ingredients || [],
                      instructions: meal.instructions || '',
                      tags: meal.tags || [],
                      mealType: meal.type,
                      date: formatDate(meal.date),
                    });
                  }
                }}
                className="w-full flex items-center gap-4 p-4 bg-white rounded-xl shadow-sm border border-gray-100 hover:border-orange-200 hover:shadow-md transition-all text-left"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <span className="text-3xl">{meal.emoji}</span>
                <div className="flex-1">
                  <p className="font-medium text-gray-800">{meal.name}</p>
                  <p className="text-sm text-gray-400">
                    {meal.type.charAt(0).toUpperCase() + meal.type.slice(1)} • {formatDate(meal.date)}
                  </p>
                </div>
                {meal.instructions && (
                  <span className="text-gray-300">→</span>
                )}
              </motion.button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center text-xs text-gray-400 mt-8 pb-4">
        Powered by <span className="font-semibold">Hindsight</span> Memory
      </footer>
    </div>
  );
}
