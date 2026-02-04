'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface OnboardingProps {
  onComplete: (username: string) => void;
  onSkip?: (username: string) => void;
  onClose?: () => void;
  initialUsername?: string | null;
}

const LANGUAGES = [
  { id: 'en', emoji: '🇬🇧', label: 'English' },
  { id: 'es', emoji: '🇪🇸', label: 'Español' },
  { id: 'fr', emoji: '🇫🇷', label: 'Français' },
  { id: 'de', emoji: '🇩🇪', label: 'Deutsch' },
  { id: 'it', emoji: '🇮🇹', label: 'Italiano' },
  { id: 'pt', emoji: '🇧🇷', label: 'Português' },
  { id: 'zh', emoji: '🇨🇳', label: '中文' },
  { id: 'ja', emoji: '🇯🇵', label: '日本語' },
  { id: 'ko', emoji: '🇰🇷', label: '한국어' },
  { id: 'hi', emoji: '🇮🇳', label: 'हिन्दी' },
];

const CUISINES = [
  { id: 'italian', emoji: '🍝', label: 'Italian' },
  { id: 'mexican', emoji: '🌮', label: 'Mexican' },
  { id: 'asian', emoji: '🍜', label: 'Asian' },
  { id: 'indian', emoji: '🍛', label: 'Indian' },
  { id: 'mediterranean', emoji: '🥙', label: 'Mediterranean' },
  { id: 'american', emoji: '🍔', label: 'American' },
  { id: 'japanese', emoji: '🍣', label: 'Japanese' },
  { id: 'thai', emoji: '🥢', label: 'Thai' },
  { id: 'french', emoji: '🥐', label: 'French' },
  { id: 'chinese', emoji: '🥡', label: 'Chinese' },
];

const DIETARY = [
  { id: 'none', emoji: '🍽️', label: 'No restrictions', description: 'I eat everything' },
  { id: 'vegetarian', emoji: '🥬', label: 'Vegetarian', description: 'No meat or fish' },
  { id: 'vegan', emoji: '🌱', label: 'Vegan', description: 'No animal products' },
  { id: 'pescatarian', emoji: '🐟', label: 'Pescatarian', description: 'Fish but no meat' },
  { id: 'keto', emoji: '🥓', label: 'Keto', description: 'Low carb, high fat' },
  { id: 'glutenfree', emoji: '🌾', label: 'Gluten-free', description: 'No gluten' },
  { id: 'dairyfree', emoji: '🥛', label: 'Dairy-free', description: 'No dairy products' },
];

const GOALS = [
  { id: 'healthy', emoji: '💪', label: 'Eat healthier' },
  { id: 'variety', emoji: '🌈', label: 'Try new foods' },
  { id: 'quick', emoji: '⚡', label: 'Quick meals' },
  { id: 'budget', emoji: '💰', label: 'Budget-friendly' },
  { id: 'mealprep', emoji: '📦', label: 'Meal prep' },
  { id: 'weightloss', emoji: '⚖️', label: 'Weight management' },
];

type Step = 'nickname' | 'language' | 'cuisines' | 'dietary' | 'goals' | 'saving';
const STEPS: Step[] = ['nickname', 'language', 'cuisines', 'dietary', 'goals'];

function getBrowserLanguage(): string {
  if (typeof navigator === 'undefined') return 'en';
  const lang = navigator.language.split('-')[0];
  return LANGUAGES.find(l => l.id === lang)?.id || 'en';
}

export default function Onboarding({ onComplete, onSkip, onClose, initialUsername }: OnboardingProps) {
  const [step, setStep] = useState<Step>('nickname');
  const [nickname, setNickname] = useState<string>(initialUsername || '');
  const [selectedLanguage, setSelectedLanguage] = useState<string>('en');
  const [selectedCuisines, setSelectedCuisines] = useState<string[]>([]);
  const [selectedDietary, setSelectedDietary] = useState<string[]>([]);
  const [selectedGoals, setSelectedGoals] = useState<string[]>([]);

  useEffect(() => {
    setSelectedLanguage(getBrowserLanguage());

    // Load existing preferences if they exist
    fetch('/api/dashboard')
      .then(res => res.json())
      .then(data => {
        if (data.preferences) {
          const prefs = data.preferences;

          // Load nickname
          if (prefs.nickname) {
            setNickname(prefs.nickname);
          }

          // Map language label back to ID
          if (prefs.language) {
            const lang = LANGUAGES.find(l => l.label === prefs.language);
            if (lang) setSelectedLanguage(lang.id);
          }

          // Map cuisine labels back to IDs
          if (prefs.cuisines?.length) {
            const cuisineIds = prefs.cuisines
              .map((label: string) => CUISINES.find(c => c.label === label)?.id)
              .filter(Boolean);
            setSelectedCuisines(cuisineIds);
          }

          // Map dietary labels back to IDs
          if (prefs.dietary?.length) {
            const dietaryIds = prefs.dietary
              .map((label: string) => DIETARY.find(d => d.label === label)?.id)
              .filter(Boolean);
            setSelectedDietary(dietaryIds);
          }

          // Map goal labels back to IDs
          if (prefs.goals?.length) {
            const goalIds = prefs.goals
              .map((label: string) => GOALS.find(g => g.label === label)?.id)
              .filter(Boolean);
            setSelectedGoals(goalIds);
          }
        }
      })
      .catch(err => console.error('Failed to load preferences:', err));
  }, []);

  const toggleCuisine = (id: string) => {
    setSelectedCuisines(prev =>
      prev.includes(id) ? prev.filter(c => c !== id) : [...prev, id]
    );
  };

  const toggleDietary = (id: string) => {
    if (id === 'none') {
      setSelectedDietary(['none']);
    } else {
      setSelectedDietary(prev => {
        const without = prev.filter(d => d !== 'none' && d !== id);
        return prev.includes(id) ? without : [...without, id];
      });
    }
  };

  const toggleGoal = (id: string) => {
    setSelectedGoals(prev =>
      prev.includes(id) ? prev.filter(g => g !== id) : [...prev, id]
    );
  };

  const goNext = () => {
    const idx = STEPS.indexOf(step);
    if (idx < STEPS.length - 1) {
      setStep(STEPS[idx + 1]);
    } else {
      savePreferences();
    }
  };

  const goBack = () => {
    const idx = STEPS.indexOf(step);
    if (idx > 0) {
      setStep(STEPS[idx - 1]);
    }
  };

  const savePreferences = async () => {
    setStep('saving');

    try {
      // Build structured preferences object for single API call
      const langLabel = LANGUAGES.find(l => l.id === selectedLanguage)?.label || 'English';

      const cuisineLabels = selectedCuisines
        .map(id => CUISINES.find(c => c.id === id)?.label)
        .filter(Boolean) as string[];

      const dietaryLabels = (selectedDietary.length > 0 && !selectedDietary.includes('none'))
        ? selectedDietary
            .map(id => DIETARY.find(d => d.id === id)?.label)
            .filter(Boolean) as string[]
        : [];

      const goalLabels = selectedGoals
        .map(id => GOALS.find(g => g.id === id)?.label)
        .filter(Boolean) as string[];

      const finalUsername = nickname.trim() || 'Guest';

      // Single API call with all preferences
      await fetch('/api/preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: finalUsername,
          preferences: {
            nickname: finalUsername,
            language: langLabel,
            cuisines: cuisineLabels,
            dietary: dietaryLabels,
            goals: goalLabels,
          },
        }),
      });

      setTimeout(() => {
        onComplete(finalUsername);
      }, 1000);
    } catch (e) {
      console.error('Failed to save preferences:', e);
      onComplete(nickname.trim() || 'Guest');
    }
  };

  const stepIndex = STEPS.indexOf(step);

  return (
    <div className="h-screen flex flex-col px-4 py-6 max-w-md mx-auto">
      <header className="mb-6">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-gray-800">
              {step === 'nickname' && "What should we call you? 👋"}
              {step === 'language' && "Choose your language 🌍"}
              {step === 'cuisines' && "What cuisines do you love? 🍽️"}
              {step === 'dietary' && "Any dietary preferences? 🥗"}
              {step === 'goals' && "What are your food goals? 🎯"}
              {step === 'saving' && "Setting up your profile..."}
            </h1>
            <p className="text-gray-500 mt-1">
              {step === 'nickname' && "This helps us personalize your experience"}
              {step === 'language' && "Recipes will be in this language"}
              {step === 'cuisines' && "Select all that apply"}
              {step === 'dietary' && "We'll personalize your suggestions"}
              {step === 'goals' && "Help us understand what you're looking for"}
              {step === 'saving' && "Just a moment"}
            </p>
          </div>
          {onClose && step !== 'saving' && (
            <button
              onClick={onClose}
              className="w-10 h-10 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 transition-colors text-gray-600"
            >
              ✕
            </button>
          )}
        </div>
      </header>

      <div className="flex gap-2 mb-6">
        {STEPS.map((s, idx) => (
          <div
            key={s}
            className={`h-1 flex-1 rounded-full transition-colors ${
              stepIndex >= idx ? 'bg-orange-500' : 'bg-gray-200'
            }`}
          />
        ))}
      </div>

      <AnimatePresence mode="wait">
        {step === 'nickname' && (
          <motion.div
            key="nickname"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1 flex flex-col justify-center"
          >
            <div className="space-y-4">
              <input
                type="text"
                value={nickname}
                onChange={(e) => setNickname(e.target.value)}
                placeholder="Enter your nickname"
                className="w-full px-4 py-3 text-lg border-2 border-gray-200 rounded-xl focus:border-orange-500 focus:outline-none transition-colors"
                autoFocus
                maxLength={30}
              />
              <p className="text-sm text-gray-400 text-center">
                We'll use this to tag your memories
              </p>
            </div>
          </motion.div>
        )}

        {step === 'language' && (
          <motion.div
            key="language"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1 overflow-y-auto"
          >
            <div className="grid grid-cols-2 gap-3">
              {LANGUAGES.map(lang => (
                <button
                  key={lang.id}
                  onClick={() => setSelectedLanguage(lang.id)}
                  className={`flex items-center gap-3 p-4 rounded-xl border-2 transition-all ${
                    selectedLanguage === lang.id
                      ? 'border-orange-500 bg-orange-50'
                      : 'border-gray-100 bg-white hover:border-gray-200'
                  }`}
                >
                  <span className="text-2xl">{lang.emoji}</span>
                  <span className={`font-medium ${
                    selectedLanguage === lang.id ? 'text-orange-700' : 'text-gray-700'
                  }`}>
                    {lang.label}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {step === 'cuisines' && (
          <motion.div
            key="cuisines"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1 overflow-y-auto"
          >
            <div className="grid grid-cols-2 gap-3">
              {CUISINES.map(cuisine => (
                <button
                  key={cuisine.id}
                  onClick={() => toggleCuisine(cuisine.id)}
                  className={`flex items-center gap-3 p-4 rounded-xl border-2 transition-all ${
                    selectedCuisines.includes(cuisine.id)
                      ? 'border-orange-500 bg-orange-50'
                      : 'border-gray-100 bg-white hover:border-gray-200'
                  }`}
                >
                  <span className="text-2xl">{cuisine.emoji}</span>
                  <span className={`font-medium ${
                    selectedCuisines.includes(cuisine.id) ? 'text-orange-700' : 'text-gray-700'
                  }`}>
                    {cuisine.label}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {step === 'dietary' && (
          <motion.div
            key="dietary"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1 overflow-y-auto"
          >
            <div className="grid grid-cols-2 gap-3">
              {DIETARY.map(diet => (
                <button
                  key={diet.id}
                  onClick={() => toggleDietary(diet.id)}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all text-center ${
                    selectedDietary.includes(diet.id)
                      ? 'border-orange-500 bg-orange-50'
                      : 'border-gray-100 bg-white hover:border-gray-200'
                  }`}
                >
                  <span className="text-2xl">{diet.emoji}</span>
                  <div>
                    <span className={`font-medium text-sm ${
                      selectedDietary.includes(diet.id) ? 'text-orange-700' : 'text-gray-700'
                    }`}>
                      {diet.label}
                    </span>
                    <p className="text-xs text-gray-500 mt-0.5">{diet.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {step === 'goals' && (
          <motion.div
            key="goals"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="flex-1 overflow-y-auto"
          >
            <div className="grid grid-cols-2 gap-3">
              {GOALS.map(goal => (
                <button
                  key={goal.id}
                  onClick={() => toggleGoal(goal.id)}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all ${
                    selectedGoals.includes(goal.id)
                      ? 'border-orange-500 bg-orange-50'
                      : 'border-gray-100 bg-white hover:border-gray-200'
                  }`}
                >
                  <span className="text-3xl">{goal.emoji}</span>
                  <span className={`font-medium text-center text-sm ${
                    selectedGoals.includes(goal.id) ? 'text-orange-700' : 'text-gray-700'
                  }`}>
                    {goal.label}
                  </span>
                </button>
              ))}
            </div>
          </motion.div>
        )}

        {step === 'saving' && (
          <motion.div
            key="saving"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex-1 flex flex-col items-center justify-center"
          >
            <div className="text-6xl mb-4">✨</div>
            <p className="text-gray-600">Saving your preferences...</p>
            <div className="mt-4 w-32 h-1 bg-gray-100 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-orange-500"
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 1 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {step !== 'saving' && (
        <div className="mt-4 flex gap-3">
          {step === 'nickname' && onSkip && (
            <button
              onClick={() => onSkip(nickname || 'Guest')}
              className="flex-1 py-3 text-gray-500 font-medium"
            >
              Skip for now
            </button>
          )}
          {step !== 'nickname' && (
            <button
              onClick={goBack}
              className="flex-1 py-3 text-gray-500 font-medium"
            >
              Back
            </button>
          )}
          <button
            onClick={goNext}
            disabled={step === 'nickname' && !nickname.trim()}
            className="flex-1 py-3 bg-gradient-to-r from-orange-500 to-rose-500 text-white font-semibold rounded-xl shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {step === 'goals' ? 'Done' : 'Next'}
          </button>
        </div>
      )}
    </div>
  );
}
