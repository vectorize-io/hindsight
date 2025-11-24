# 🚀 AI Fitness Coach - Quick Start

Your AI fitness coach is ready! Here's how to start using it.

## ✅ What's Done

1. **Fitness coach agent created** with supportive personality traits
2. **Workout logging** script ready
3. **Meal logging** script ready
4. **Coach chat interface** ready
5. **Demo script** to see everything in action

## 🎯 Quick Test (5 minutes)

### Step 1: Log a Workout

```bash
cd fitness-coach

# Quick command-line log
python log_workout.py strength 45 squats deadlifts "bench press"

# Or interactive mode
python log_workout.py
```

### Step 2: Log a Meal

```bash
# Quick command-line log
python log_meal.py lunch "chicken breast" "brown rice" broccoli

# Or interactive mode
python log_meal.py
```

### Step 3: Chat with Your Coach

```bash
# Ask a question
python coach_chat.py "What have I been doing for exercise?"

# Or start interactive chat
python coach_chat.py
```

## 🎬 Full Demo

Run the complete demo to see all features:

```bash
python demo.py
```

This will:
1. Log 4 sample workouts (cardio, strength, yoga)
2. Log 4 sample meals with nutrition info
3. Ask the coach 4 different questions
4. Show how the coach forms opinions about your habits

## 📂 Files Created

All files are in `/fitness-coach/`:

- `setup_coach.py` - Creates the fitness coach agent ✅ (already run)
- `log_workout.py` - Log workouts
- `log_meal.py` - Log meals
- `coach_chat.py` - Chat with your coach
- `demo.py` - Complete demo with sample data
- `README.md` - Full documentation

## 🔥 Example Questions to Ask

Try asking your coach:

```bash
python coach_chat.py "What workouts did I do this week?"
python coach_chat.py "How is my nutrition?"
python coach_chat.py "Should I take a rest day?"
python coach_chat.py "What should I focus on next?"
python coach_chat.py "Am I getting enough protein?"
```

## 🎨 How It Works

1. **Log Data**: Workouts and meals are stored as "world facts" with timestamps
2. **Memory System**: Entity linking, temporal queries, semantic search
3. **Coach Thinks**: Uses the `/think` API to retrieve relevant history
4. **Personalized Advice**: Based on your actual data + coach personality
5. **Opinion Formation**: Coach forms beliefs about your habits (confidence scores)

## 💡 Tips

- Log consistently for best results
- Be specific with exercise names and foods
- Ask questions about trends and progress
- The coach learns from your interactions

## 🏗️ System Status

- ✅ API running at http://localhost:8080
- ✅ PostgreSQL running at localhost:5432
- ✅ Fitness coach agent created
- ✅ All scripts tested and ready

## 📖 Need More Help?

See `README.md` for full documentation including:
- Architecture details
- All command-line options
- More example interactions
- How the memory system works

---

**Ready to start? Run the demo:**

```bash
python demo.py
```

Or jump right in:

```bash
python log_workout.py  # Log your first workout!
```
