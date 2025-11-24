# 🏋️ AI Fitness Coach

A personalized AI fitness coach that learns from your workouts and meals to provide customized advice and motivation.

## Features

- **Personalized Coaching**: AI coach with personality traits (supportive, disciplined, motivating)
- **Workout Tracking**: Log cardio, strength training, yoga, and custom workouts
- **Meal Tracking**: Log meals with nutrition information
- **Smart Advice**: Ask questions and get context-aware coaching based on your history
- **Progressive Learning**: Coach forms opinions about your habits and progress over time

## Quick Start

### 1. Setup the Coach

First, create the fitness coach agent:

```bash
python setup_coach.py
```

This creates an AI coach with:
- High conscientiousness (disciplined, organized)
- High agreeableness (supportive, encouraging)
- Moderate-high extraversion (energetic, motivating)
- Low neuroticism (calm, stable)

### 2. Run the Demo

See everything in action:

```bash
python demo.py
```

The demo will:
1. Log sample workouts (running, strength training, yoga)
2. Log sample meals with nutrition info
3. Ask the coach questions and show personalized responses

## Usage

### Log Workouts

**Interactive mode:**
```bash
python log_workout.py
```

**Command-line:**
```bash
# Cardio workout
python log_workout.py cardio 30 5  # 30 min, 5 km

# Strength training
python log_workout.py strength 45 squats deadlifts "bench press"

# Yoga
python log_workout.py yoga 30
```

### Log Meals

**Interactive mode:**
```bash
python log_meal.py
```

**Command-line:**
```bash
python log_meal.py breakfast oatmeal banana "protein shake"
python log_meal.py lunch "chicken breast" "brown rice" broccoli
python log_meal.py dinner salmon quinoa vegetables
```

### Chat with Your Coach

**Interactive mode:**
```bash
python coach_chat.py
```

**Single question:**
```bash
python coach_chat.py "Should I take a rest day tomorrow?"
python coach_chat.py "How is my nutrition?"
python coach_chat.py "What should I focus on this week?"
```

## Example Questions to Ask

- "What have I been doing for exercise?"
- "How is my nutrition looking?"
- "Should I take a rest day tomorrow?"
- "Am I getting enough protein?"
- "What workouts did I do last week?"
- "What should I focus on to meet my goals?"
- "How has my performance been improving?"

## How It Works

### Memory System

The coach uses a temporal-semantic memory system with three types of memories:

1. **World Facts**: Your actual workout and meal data
   - "User completed 30-minute cardio workout..."
   - "User ate chicken breast with brown rice..."

2. **Agent Facts**: Your goals and intentions
   - "User wants to build strength"
   - "User is training for a 5K"

3. **Opinions**: Coach's assessments and insights
   - "User is consistent with morning workouts [0.9]"
   - "User needs more recovery days [0.7]"

### Personalization

The coach's advice is personalized based on:
- Your workout history (types, intensity, frequency)
- Your meal patterns (nutrition, timing)
- Temporal patterns (what you did last week/month)
- Formed opinions about your habits and progress
- The coach's personality traits

### Example Interaction

```
You: Should I take a rest day tomorrow?

🤔 Coach is thinking...

======================================================================
🏋️ COACH'S ADVICE
======================================================================

Based on your recent activity, I'd recommend taking a rest day tomorrow.
You've had 4 solid workout days this week including a high-intensity leg
day where you hit a new PR on squats - that's fantastic! Your body needs
time to recover and rebuild. Consider doing some light stretching or yoga
if you feel restless, but give those muscles a chance to adapt. Remember,
rest is when the real gains happen!

----------------------------------------------------------------------
📊 BASED ON:
----------------------------------------------------------------------

🌍 [WORLD]
   User completed 45-minute strength workout with high intensity.
   Exercises: squats, deadlifts, bench press. Notes: Leg day, hit new PR!
   📅 2025-01-20

🌍 [WORLD]
   User completed 30-minute cardio workout with moderate intensity.
   📅 2025-01-19

💭 [OPINION]
   User is very consistent with workout schedule [0.85]

----------------------------------------------------------------------
✨ NEW INSIGHTS FORMED:
----------------------------------------------------------------------

💡 User responds well to recovery guidance
   Confidence: 75%

======================================================================
```

## Architecture

Built on Memora's temporal-semantic memory system:
- Entity linking connects workouts to goals
- Temporal queries track progress over time
- Personality-driven opinion formation
- Multi-strategy retrieval (semantic + keyword + graph + temporal)

## Next Steps

1. **Add Goals**: Store your fitness goals as agent facts
2. **Track Progress**: Log measurements (weight, strength PRs, distances)
3. **Build History**: Consistent logging helps the coach learn your patterns
4. **Ask Questions**: The more you interact, the better the advice becomes

## Tips

- Log workouts and meals consistently for best results
- Be specific with exercise names and food details
- Ask questions about trends ("How has my progress been?")
- Use temporal queries ("What did I do last week?")
- Let the coach form opinions by using the chat regularly
