'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Dashboard from '@/components/Dashboard';
import EatNow from '@/components/EatNow';
import Onboarding from '@/components/Onboarding';
import RecipeView, { Recipe } from '@/components/RecipeView';
import { getStoredUsername, setStoredUsername } from '@/lib/storage';

export type Screen = 'onboarding' | 'dashboard' | 'eat-now' | 'preferences' | 'recipe';

export default function Home() {
  const [screen, setScreen] = useState<Screen | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [currentRecipe, setCurrentRecipe] = useState<Recipe | null>(null);
  const [recipeMode, setRecipeMode] = useState<'cooking' | 'viewing'>('viewing');

  useEffect(() => {
    // Check localStorage for cached username
    const storedUsername = getStoredUsername();

    if (storedUsername) {
      setUsername(storedUsername);
      // Verify user exists by loading their document
      fetch(`/api/dashboard?username=${encodeURIComponent(storedUsername)}`)
        .then(res => res.json())
        .then(data => {
          const prefs = data.preferences || {};
          const hasPrefs = !!(prefs.nickname);
          setScreen(hasPrefs ? 'dashboard' : 'onboarding');
        })
        .catch(() => {
          setScreen('onboarding');
        });
    } else {
      setScreen('onboarding');
    }
  }, []);

  const handleMealLogged = (recipe?: Recipe) => {
    if (recipe) {
      setCurrentRecipe(recipe);
      setRecipeMode('cooking');
      setScreen('recipe');
    } else {
      setRefreshKey(k => k + 1);
      setScreen('dashboard');
    }
  };

  const handleRecipeClose = () => {
    setRefreshKey(k => k + 1);
    setScreen('dashboard');
    setCurrentRecipe(null);
  };

  const handleViewRecipe = (recipe: Recipe) => {
    setCurrentRecipe(recipe);
    setRecipeMode('viewing');
    setScreen('recipe');
  };

  const handleOnboardingComplete = (newUsername: string) => {
    setUsername(newUsername);
    setStoredUsername(newUsername);
    setScreen('dashboard');
  };

  if (!screen) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-rose-50 flex items-center justify-center">
        <div className="text-4xl animate-pulse">🍳</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-rose-50">
      <AnimatePresence mode="wait">
        {screen === 'onboarding' && (
          <motion.div
            key="onboarding"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Onboarding
              onComplete={handleOnboardingComplete}
              onSkip={(nickname) => handleOnboardingComplete(nickname || 'Guest')}
              initialUsername={username}
            />
          </motion.div>
        )}
        {screen === 'preferences' && (
          <motion.div
            key="preferences"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
            <Onboarding
              onComplete={(newUsername) => {
                setUsername(newUsername);
                setStoredUsername(newUsername);
                setScreen('dashboard');
              }}
              onClose={() => setScreen('dashboard')}
              initialUsername={username}
            />
          </motion.div>
        )}
        {screen === 'dashboard' && username && (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Dashboard
              username={username}
              onEatNow={() => setScreen('eat-now')}
              onOpenPreferences={() => setScreen('preferences')}
              onViewRecipe={handleViewRecipe}
              refreshKey={refreshKey}
            />
          </motion.div>
        )}
        {screen === 'eat-now' && username && (
          <motion.div
            key="eat-now"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
          >
            <EatNow
              username={username}
              onBack={() => setScreen('dashboard')}
              onMealLogged={handleMealLogged}
            />
          </motion.div>
        )}
        {screen === 'recipe' && currentRecipe && (
          <motion.div
            key="recipe"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
          >
            <RecipeView
              recipe={currentRecipe}
              onClose={handleRecipeClose}
              showCookingMode={recipeMode === 'cooking'}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
