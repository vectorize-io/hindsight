'use client';

import { motion } from 'framer-motion';

export interface Recipe {
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

interface RecipeViewProps {
  recipe: Recipe;
  onClose: () => void;
  showCookingMode?: boolean;
}

export default function RecipeView({ recipe, onClose, showCookingMode }: RecipeViewProps) {
  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${minutes} min`;
    const hrs = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hrs}h ${mins}m` : `${hrs}h`;
  };

  return (
    <div className="h-screen flex flex-col px-4 py-4 max-w-md mx-auto">
      {/* Header */}
      <header className="flex items-center gap-4 mb-4 flex-shrink-0">
        <button
          onClick={onClose}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-white shadow-sm border border-gray-100"
        >
          ←
        </button>
        <div>
          <h1 className="text-xl font-bold text-gray-800">
            {showCookingMode ? "Let's cook! 👨‍🍳" : 'Recipe'}
          </h1>
          {recipe.date && (
            <p className="text-sm text-gray-500">{recipe.date}</p>
          )}
        </div>
      </header>

      {/* Recipe Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex-1 overflow-y-auto bg-white rounded-2xl shadow-lg border border-gray-100 p-5"
      >
        {/* Header */}
        <div className="flex items-start gap-4 mb-4">
          <span className="text-6xl">{recipe.emoji}</span>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-gray-800">{recipe.name}</h2>
            {recipe.description && (
              <p className="text-gray-500 mt-1">{recipe.description}</p>
            )}
            <div className="flex items-center gap-4 mt-3">
              <span className="flex items-center gap-1 text-sm text-gray-600">
                <span>⏱️</span> {formatTime(recipe.timeMinutes)}
              </span>
              <span className={`text-sm font-semibold ${
                recipe.healthScore >= 7 ? 'text-green-500' :
                recipe.healthScore >= 5 ? 'text-yellow-600' : 'text-orange-500'
              }`}>
                {recipe.healthScore}/10 health
              </span>
            </div>
          </div>
        </div>

        {/* Tags */}
        {recipe.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-5">
            {recipe.tags.map(tag => (
              <span
                key={tag}
                className="px-3 py-1 bg-gray-100 text-gray-600 text-sm rounded-full"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Ingredients */}
        <div className="mb-5">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">🛒 Ingredients</h3>
          <div className="grid grid-cols-2 gap-2">
            {recipe.ingredients.map((ing, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 p-2 bg-orange-50 rounded-lg border border-orange-100"
              >
                <span className="w-2 h-2 bg-orange-400 rounded-full flex-shrink-0" />
                <span className="text-sm text-gray-700">{ing}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Instructions */}
        <div className="mb-5">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">👨‍🍳 Instructions</h3>
          <div className="bg-gray-50 rounded-xl p-4">
            <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
              {recipe.instructions}
            </p>
          </div>
        </div>

        {showCookingMode && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-center py-4"
          >
            <p className="text-gray-500 text-sm">Enjoy your meal! 🍽️</p>
          </motion.div>
        )}
      </motion.div>

      {/* Done Button */}
      <div className="mt-4 flex-shrink-0">
        <button
          onClick={onClose}
          className="w-full py-4 bg-gradient-to-r from-orange-500 to-rose-500 text-white text-lg font-semibold rounded-2xl shadow-lg"
        >
          {showCookingMode ? '✓ Done cooking' : 'Back'}
        </button>
      </div>
    </div>
  );
}
