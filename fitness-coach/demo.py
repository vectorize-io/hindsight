#!/usr/bin/env python3
"""
Fitness Coach Demo
Demonstrates the complete AI Fitness Coach functionality with sample data.
"""
import time
from log_workout import log_workout
from log_meal import log_meal
from coach_chat import ask_coach, print_response

def demo():
    """Run a complete demo of the fitness coach."""

    print("\n" + "=" * 70)
    print("🏋️ AI FITNESS COACH DEMO")
    print("=" * 70)
    print("\nThis demo will:")
    print("  1. Log sample workouts and meals")
    print("  2. Ask the coach personalized questions")
    print("  3. Show how the coach learns and forms opinions")
    print("\n" + "=" * 70 + "\n")

    input("Press Enter to start the demo...")

    # Step 1: Log workouts
    print("\n📝 Step 1: Logging Workouts...")
    print("-" * 70)

    workouts = [
        ("cardio", 30, ["running"], "moderate", "Morning 5K run, felt good"),
        ("strength", 45, ["squats", "deadlifts", "bench press"], "high", "Leg day, hit new PR on squats!"),
        ("yoga", 30, ["sun salutations", "warrior poses"], "low", "Recovery day stretching"),
        ("cardio", 25, ["cycling"], "high", "HIIT cycling session"),
    ]

    for workout in workouts:
        workout_type, duration, exercises, intensity, notes = workout
        log_workout(workout_type, duration, exercises, intensity, notes)
        time.sleep(0.5)

    print("\n✅ Workouts logged!\n")
    input("Press Enter to continue...")

    # Step 2: Log meals
    print("\n📝 Step 2: Logging Meals...")
    print("-" * 70)

    meals = [
        ("breakfast", ["oatmeal", "banana", "protein shake"], 450, 35, 55, 12, "Post-workout breakfast"),
        ("lunch", ["chicken breast", "brown rice", "broccoli"], 550, 45, 60, 15, "Balanced lunch"),
        ("snack", ["apple", "almonds"], 200, 5, 25, 15, "Afternoon snack"),
        ("dinner", ["salmon", "quinoa", "mixed vegetables"], 600, 40, 50, 25, "Light dinner"),
    ]

    for meal in meals:
        meal_type, foods, calories, protein, carbs, fats, notes = meal
        log_meal(meal_type, foods, calories, protein, carbs, fats, notes)
        time.sleep(0.5)

    print("\n✅ Meals logged!\n")
    input("Press Enter to continue...")

    # Step 3: Ask the coach questions
    print("\n💬 Step 3: Asking the Coach Questions...")
    print("-" * 70 + "\n")

    questions = [
        "What have I been doing for exercise?",
        "How is my nutrition looking?",
        "Should I take a rest day tomorrow?",
        "What do you recommend I focus on next?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'=' * 70}")
        print(f"\nYou: {question}")
        print("\n🤔 Coach is thinking...")

        result = ask_coach(question)
        if result:
            print_response(result)

        if i < len(questions):
            input("\nPress Enter for next question...")

    # Final message
    print("\n" + "=" * 70)
    print("✨ DEMO COMPLETE!")
    print("=" * 70)
    print("\nThe fitness coach now knows about your:")
    print("  ✅ Workout history (types, duration, intensity)")
    print("  ✅ Meal history (foods, nutrition, timing)")
    print("  ✅ Progress patterns and habits")
    print("  ✅ Personalized coaching insights")
    print("\nYou can now use:")
    print("  • python log_workout.py  - Log new workouts")
    print("  • python log_meal.py     - Log new meals")
    print("  • python coach_chat.py   - Chat with your coach")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for watching!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
