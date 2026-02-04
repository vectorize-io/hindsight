'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence, PanInfo } from 'framer-motion';

interface Recipe {
  name: string;
  emoji: string;
  description?: string;
  healthScore: number;
  timeMinutes: number;
  ingredients: string[];
  instructions: string;
  tags: string[];
}

interface EatNowProps {
  username: string;
  onBack: () => void;
  onMealLogged: (recipe?: Recipe) => void;
}

interface FoodOption {
  id: string;
  name: string;
  emoji: string;
  description: string;
  healthScore: number;
  timeMinutes: number;
  ingredients: string[];
  instructions: string;
  tags: string[];
}

type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';

function getSuggestedMealType(): MealType {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 11) return 'breakfast';
  if (hour >= 11 && hour < 15) return 'lunch';
  if (hour >= 15 && hour < 18) return 'snack';
  return 'dinner';
}

type CookingTime = 'none' | 'quick' | 'medium' | 'long' | 'extended';

const cookingTimes: { value: CookingTime; emoji: string; label: string; description: string }[] = [
  { value: 'none', emoji: '🥣', label: 'No cooking', description: 'Ready to eat' },
  { value: 'quick', emoji: '⚡', label: '< 5 min', description: 'Super quick' },
  { value: 'medium', emoji: '🍳', label: '5-15 min', description: 'Quick meal' },
  { value: 'long', emoji: '🥘', label: '15-30 min', description: 'Normal cooking' },
  { value: 'extended', emoji: '👨‍🍳', label: '30+ min', description: 'Full recipe' },
];

export default function EatNow({ username, onBack, onMealLogged }: EatNowProps) {
  const [step, setStep] = useState<'type' | 'time' | 'options' | 'logging'>('type');
  const [mealType, setMealType] = useState<MealType | null>(null);
  const [cookingTime, setCookingTime] = useState<CookingTime | null>(null);
  const suggestedMeal = getSuggestedMealType();
  const [options, setOptions] = useState<FoodOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFood, setSelectedFood] = useState<FoodOption | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [direction, setDirection] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [insights, setInsights] = useState<string[]>([]);

  const mealTypes: { type: MealType; emoji: string; label: string }[] = [
    { type: 'breakfast', emoji: '🌅', label: 'Breakfast' },
    { type: 'lunch', emoji: '☀️', label: 'Lunch' },
    { type: 'dinner', emoji: '🌙', label: 'Dinner' },
    { type: 'snack', emoji: '🍿', label: 'Snack' },
  ];

  const selectMealType = (type: MealType) => {
    setMealType(type);
    setStep('time');
  };

  const selectCookingTime = async (time: CookingTime) => {
    setCookingTime(time);
    setLoading(true);
    setStep('options');
    setCurrentIndex(0);
    setError(null);

    try {
      const res = await fetch('/api/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, mealType, cookingTime: time }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setOptions([]);
        setInsights([]);
      } else {
        setOptions(data.suggestions || []);
        setInsights(data.insights || []);
      }
    } catch (e) {
      console.error('Failed to get suggestions:', e);
      setError('Failed to connect to the server. Please try again.');
      setOptions([]);
      setInsights([]);
    }
    setLoading(false);
  };

  const goNext = useCallback(() => {
    if (currentIndex < options.length - 1) {
      setDirection(1);
      setCurrentIndex(prev => prev + 1);
    }
  }, [currentIndex, options.length]);

  const goPrev = useCallback(() => {
    if (currentIndex > 0) {
      setDirection(-1);
      setCurrentIndex(prev => prev - 1);
    }
  }, [currentIndex]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (step !== 'options' || loading) return;

      if (e.key === 'ArrowRight') {
        goNext();
      } else if (e.key === 'ArrowLeft') {
        goPrev();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [step, loading, goNext, goPrev]);

  // Swipe handler
  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    const swipeThreshold = 50;

    if (info.offset.x < -swipeThreshold) {
      goNext();
    } else if (info.offset.x > swipeThreshold) {
      goPrev();
    }
  };

  const handleAction = async (food: FoodOption, action: 'ate_today' | 'ate_yesterday' | 'never' | 'cook') => {
    setActionLoading(food.id);

    try {
      await fetch('/api/log-meal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username,
          food,
          action,
          mealType,
        }),
      });

      if (action === 'cook') {
        // Pass recipe to show cooking view
        onMealLogged({
          name: food.name,
          emoji: food.emoji,
          description: food.description,
          healthScore: food.healthScore,
          timeMinutes: food.timeMinutes,
          ingredients: food.ingredients,
          instructions: food.instructions,
          tags: food.tags,
        });
      } else if (action === 'never') {
        const newOptions = options.filter(f => f.id !== food.id);
        setOptions(newOptions);
        if (currentIndex >= newOptions.length && newOptions.length > 0) {
          setCurrentIndex(newOptions.length - 1);
        }
      } else {
        const newOptions = options.filter(f => f.id !== food.id);
        setOptions(newOptions);
        if (currentIndex >= newOptions.length && newOptions.length > 0) {
          setCurrentIndex(newOptions.length - 1);
        }
      }
    } catch (e) {
      console.error('Failed to log:', e);
    }

    setActionLoading(null);
  };

  const currentFood = options[currentIndex];

  const slideVariants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 200 : -200,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (direction: number) => ({
      x: direction < 0 ? 200 : -200,
      opacity: 0,
    }),
  };

  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${minutes} min`;
    const hrs = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return mins > 0 ? `${hrs}h ${mins}m` : `${hrs}h`;
  };

  return (
    <div className="h-screen flex flex-col px-4 py-4 max-w-md mx-auto overflow-hidden">
      {/* Header */}
      <header className="flex items-center gap-4 mb-3 flex-shrink-0">
        <button
          onClick={onBack}
          className="w-10 h-10 flex items-center justify-center rounded-full bg-white shadow-sm border border-gray-100"
        >
          ←
        </button>
        <div>
          <h1 className="text-xl font-bold text-gray-800">
            {step === 'type' && 'What meal is this?'}
            {step === 'time' && 'How much time do you have?'}
            {step === 'options' && `${mealType?.charAt(0).toUpperCase()}${mealType?.slice(1)} Ideas`}
            {step === 'logging' && 'Great choice!'}
          </h1>
        </div>
      </header>

      <AnimatePresence mode="wait">
        {/* Step 1: Meal Type Selection */}
        {step === 'type' && (
          <motion.div
            key="type"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="grid grid-cols-2 gap-4 flex-1 content-start"
          >
            {mealTypes.map((meal, idx) => {
              const isSuggested = meal.type === suggestedMeal;
              return (
                <motion.button
                  key={meal.type}
                  onClick={() => selectMealType(meal.type)}
                  className={`flex flex-col items-center gap-2 p-6 rounded-2xl shadow-sm border transition-all ${
                    isSuggested
                      ? 'bg-orange-50 border-orange-300 ring-2 ring-orange-200'
                      : 'bg-white border-gray-100 hover:border-orange-200 hover:shadow-md'
                  }`}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.05, duration: 0.15 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <span className="text-4xl">{meal.emoji}</span>
                  <span className={`font-medium ${isSuggested ? 'text-orange-700' : 'text-gray-700'}`}>
                    {meal.label}
                  </span>
                  {isSuggested && (
                    <span className="text-xs text-orange-500 font-medium">Suggested</span>
                  )}
                </motion.button>
              );
            })}
          </motion.div>
        )}

        {/* Step 2: Cooking Time Selection */}
        {step === 'time' && (
          <motion.div
            key="time"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="space-y-3"
          >
            {cookingTimes.map((time, idx) => (
              <motion.button
                key={time.value}
                onClick={() => selectCookingTime(time.value)}
                className="w-full flex items-center gap-4 p-4 bg-white rounded-xl shadow-sm border border-gray-100 hover:border-orange-200 hover:shadow-md transition-all text-left"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05, duration: 0.15 }}
                whileTap={{ scale: 0.98 }}
              >
                <span className="text-3xl">{time.emoji}</span>
                <div>
                  <span className="font-semibold text-gray-800">{time.label}</span>
                  <p className="text-sm text-gray-500">{time.description}</p>
                </div>
              </motion.button>
            ))}
          </motion.div>
        )}

        {/* Step 3: Food Options - Swipeable Card */}
        {step === 'options' && (
          <motion.div
            key="options"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="flex flex-col flex-1 min-h-0"
          >
            {loading ? (
              <div className="flex-1 flex flex-col items-center justify-center">
                {/* Animated cooking emojis */}
                <div className="flex items-center gap-2 mb-6">
                  <motion.span
                    className="text-4xl"
                    animate={{ rotate: [0, -10, 10, -10, 0], y: [0, -5, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, repeatDelay: 0.3 }}
                  >
                    🍳
                  </motion.span>
                  <motion.span
                    className="text-4xl"
                    animate={{ scale: [1, 1.2, 1], rotate: [0, 5, -5, 0] }}
                    transition={{ duration: 0.8, repeat: Infinity, delay: 0.2 }}
                  >
                    👨‍🍳
                  </motion.span>
                  <motion.span
                    className="text-4xl"
                    animate={{ rotate: [0, 15, -15, 0], y: [0, -3, 0] }}
                    transition={{ duration: 0.7, repeat: Infinity, delay: 0.4 }}
                  >
                    ✨
                  </motion.span>
                </div>

                {/* Fun loading messages */}
                <motion.p
                  className="text-lg font-semibold text-gray-700 mb-2"
                  animate={{ opacity: [1, 0.7, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  Let me cook...
                </motion.p>
                <p className="text-sm text-gray-500 text-center px-8">
                  Crafting personalized recipes based on your taste
                </p>

                {/* Progress bar */}
                <div className="w-48 h-1.5 bg-gray-200 rounded-full mt-6 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-orange-500 to-rose-500 rounded-full"
                    initial={{ width: '0%' }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 3, ease: 'easeInOut' }}
                  />
                </div>
              </div>
            ) : error ? (
              <div className="text-center py-12">
                <p className="text-4xl mb-4">😕</p>
                <p className="text-gray-700 font-medium mb-2">Oops!</p>
                <p className="text-gray-500 text-sm mb-4">{error}</p>
                <button
                  onClick={() => selectCookingTime(cookingTime!)}
                  className="px-6 py-2 bg-orange-500 text-white rounded-full font-medium hover:bg-orange-600 transition-colors"
                >
                  Try again
                </button>
              </div>
            ) : options.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-4xl mb-4">🤔</p>
                <p className="text-gray-500">No more suggestions!</p>
                <button
                  onClick={() => selectCookingTime(cookingTime!)}
                  className="mt-4 px-6 py-2 bg-orange-100 text-orange-600 rounded-full font-medium"
                >
                  Get more ideas
                </button>
              </div>
            ) : (
              <div className="flex flex-col flex-1 min-h-0">
                {/* Memory Insights */}
                {insights.length > 0 && (
                  <div className="mb-3 p-3 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-100 flex-shrink-0">
                    <div className="text-xs font-semibold text-purple-700 mb-1.5 flex items-center gap-1">
                      <span>✨</span> Memory Insights
                    </div>
                    <div className="space-y-1">
                      {insights.slice(0, 3).map((insight, idx) => (
                        <p key={idx} className="text-xs text-gray-600 leading-relaxed">
                          {insight}
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                {/* Instructions */}
                <div className="text-center text-xs text-gray-400 mb-2 flex-shrink-0">
                  <span className="hidden sm:inline">← → arrow keys to browse</span>
                  <span className="sm:hidden">Swipe to browse</span>
                  {' • '}{currentIndex + 1}/{options.length}
                </div>

                {/* Swipeable Card Container */}
                <div className="relative w-full flex-1 min-h-0">
                  <AnimatePresence initial={false} custom={direction} mode="wait">
                    <motion.div
                      key={currentFood.id}
                      custom={direction}
                      variants={slideVariants}
                      initial="enter"
                      animate="center"
                      exit="exit"
                      transition={{ type: 'tween', duration: 0.2 }}
                      drag="x"
                      dragConstraints={{ left: -100, right: 100 }}
                      dragElastic={0.5}
                      onDragEnd={handleDragEnd}
                      className="absolute inset-0 bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden cursor-grab active:cursor-grabbing"
                    >
                      {/* Food Card Content */}
                      <div className="flex flex-col h-full p-4 overflow-y-auto">
                        {/* Header: Emoji, Name, Time, Health */}
                        <div className="flex items-start gap-3 mb-3">
                          <span className="text-5xl">{currentFood.emoji}</span>
                          <div className="flex-1 min-w-0">
                            <h2 className="text-xl font-bold text-gray-800 leading-tight">{currentFood.name}</h2>
                            <p className="text-gray-500 text-sm mt-1">{currentFood.description}</p>
                            <div className="flex items-center gap-3 mt-2">
                              <span className="flex items-center gap-1 text-sm text-gray-600">
                                <span>⏱️</span> {formatTime(currentFood.timeMinutes)}
                              </span>
                              <span className={`text-sm font-semibold ${
                                currentFood.healthScore >= 7 ? 'text-green-500' :
                                currentFood.healthScore >= 5 ? 'text-yellow-600' : 'text-orange-500'
                              }`}>
                                {currentFood.healthScore}/10 health
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Tags */}
                        <div className="flex flex-wrap gap-1.5 mb-3">
                          {currentFood.tags.map(tag => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>

                        {/* Ingredients */}
                        <div className="mb-3">
                          <h3 className="text-sm font-semibold text-gray-700 mb-1.5">Ingredients</h3>
                          <div className="flex flex-wrap gap-1.5">
                            {currentFood.ingredients.map((ing, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 bg-orange-50 text-orange-700 text-xs rounded-lg border border-orange-100"
                              >
                                {ing}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Instructions */}
                        <div className="mb-3 flex-1">
                          <h3 className="text-sm font-semibold text-gray-700 mb-1.5">How to make</h3>
                          <p className="text-sm text-gray-600 leading-relaxed">
                            {currentFood.instructions}
                          </p>
                        </div>

                        {/* Action Buttons */}
                        <div className="mt-auto space-y-2 pt-2">
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleAction(currentFood, 'ate_today')}
                              disabled={actionLoading === currentFood.id}
                              className="flex-1 py-2.5 px-3 bg-blue-50 text-blue-600 text-sm font-medium rounded-xl hover:bg-blue-100 transition-colors disabled:opacity-50"
                            >
                              Ate today
                            </button>
                            <button
                              onClick={() => handleAction(currentFood, 'ate_yesterday')}
                              disabled={actionLoading === currentFood.id}
                              className="flex-1 py-2.5 px-3 bg-gray-50 text-gray-600 text-sm font-medium rounded-xl hover:bg-gray-100 transition-colors disabled:opacity-50"
                            >
                              Yesterday
                            </button>
                            <button
                              onClick={() => handleAction(currentFood, 'never')}
                              disabled={actionLoading === currentFood.id}
                              className="py-2.5 px-3 bg-red-50 text-red-500 text-sm font-medium rounded-xl hover:bg-red-100 transition-colors disabled:opacity-50"
                              title="Never suggest this again"
                            >
                              Never
                            </button>
                          </div>
                          <button
                            onClick={() => handleAction(currentFood, 'cook')}
                            disabled={actionLoading === currentFood.id}
                            className="w-full py-3.5 bg-gradient-to-r from-orange-500 to-rose-500 text-white text-lg font-semibold rounded-xl shadow-md hover:shadow-lg transition-shadow disabled:opacity-50"
                          >
                            {actionLoading === currentFood.id ? '...' : "🍳 Let's cook this!"}
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  </AnimatePresence>

                  {/* Navigation Arrows (Desktop) */}
                  <button
                    onClick={goPrev}
                    disabled={currentIndex === 0}
                    className="hidden sm:flex absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-10 h-10 bg-white rounded-full shadow-lg items-center justify-center text-gray-600 hover:bg-gray-50 disabled:opacity-30 disabled:cursor-not-allowed z-10"
                  >
                    ←
                  </button>
                  <button
                    onClick={goNext}
                    disabled={currentIndex === options.length - 1}
                    className="hidden sm:flex absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-10 h-10 bg-white rounded-full shadow-lg items-center justify-center text-gray-600 hover:bg-gray-50 disabled:opacity-30 disabled:cursor-not-allowed z-10"
                  >
                    →
                  </button>
                </div>

                {/* Pagination Dots */}
                <div className="flex justify-center gap-1.5 mt-2 flex-shrink-0">
                  {options.map((_, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setDirection(idx > currentIndex ? 1 : -1);
                        setCurrentIndex(idx);
                      }}
                      className={`w-1.5 h-1.5 rounded-full transition-all ${
                        idx === currentIndex
                          ? 'w-4 bg-orange-500'
                          : 'bg-gray-300 hover:bg-gray-400'
                      }`}
                    />
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Step 3: Logging Confirmation */}
        {step === 'logging' && selectedFood && (
          <motion.div
            key="logging"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.2 }}
            className="text-center py-12"
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', duration: 0.3 }}
              className="text-7xl mb-4"
            >
              {selectedFood.emoji}
            </motion.div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Enjoy your {selectedFood.name}!
            </h2>
            <p className="text-gray-500">
              Logged to your food memory
            </p>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '100%' }}
              transition={{ duration: 1.5 }}
              className="h-1 bg-gradient-to-r from-orange-500 to-rose-500 rounded-full mt-6 mx-auto max-w-xs"
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
