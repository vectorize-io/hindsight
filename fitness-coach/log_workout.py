#!/usr/bin/env python3
"""
Workout logging script for the AI Fitness Coach.
Allows logging workouts with details like type, duration, exercises, etc.
"""
import requests
import json
from datetime import datetime
import sys

API_URL = "http://localhost:8080/api/v1"
AGENT_ID = "fitness-coach"

def log_workout(workout_type, duration_minutes, exercises=None, intensity="moderate", notes=""):
    """
    Log a workout session.

    Args:
        workout_type: Type of workout (e.g., "cardio", "strength", "yoga")
        duration_minutes: Duration in minutes
        exercises: List of exercises performed
        intensity: Intensity level ("low", "moderate", "high")
        notes: Additional notes about the workout
    """

    # Build workout description
    exercises_str = ", ".join(exercises) if exercises else "general exercises"

    content = f"User completed a {duration_minutes}-minute {workout_type} workout with {intensity} intensity. "
    content += f"Exercises: {exercises_str}. "
    if notes:
        content += f"Notes: {notes}"

    # Prepare data for API
    timestamp = datetime.now().isoformat()

    payload = {
        "agent_id": AGENT_ID,
        "items": [{
            "content": content,
            "context": f"workout-{workout_type}",
            "event_date": timestamp
        }]
    }

    # Send to API
    response = requests.post(
        f"{API_URL}/agents/{AGENT_ID}/memories",
        json=payload
    )

    if response.status_code == 200:
        print(f"✅ Workout logged successfully!")
        print(f"   Type: {workout_type}")
        print(f"   Duration: {duration_minutes} min")
        print(f"   Intensity: {intensity}")
        print(f"   Exercises: {exercises_str}")
        return True
    else:
        print(f"❌ Error logging workout: {response.status_code}")
        print(response.text)
        return False

def quick_log_cardio(duration, distance=None):
    """Quick log for cardio workouts."""
    notes = f"Distance: {distance} km" if distance else ""
    exercises = ["running"] if "run" in notes.lower() else ["cardio"]
    return log_workout("cardio", duration, exercises, intensity="moderate", notes=notes)

def quick_log_strength(duration, exercises):
    """Quick log for strength training."""
    return log_workout("strength", duration, exercises, intensity="high")

def quick_log_yoga(duration):
    """Quick log for yoga/stretching."""
    return log_workout("yoga", duration, ["yoga poses", "stretching"], intensity="low")

def interactive_log():
    """Interactive workout logging."""
    print("🏋️ Workout Logger")
    print("=" * 50)

    workout_type = input("Workout type (cardio/strength/yoga/other): ").strip().lower()
    duration = int(input("Duration (minutes): ").strip())
    intensity = input("Intensity (low/moderate/high) [moderate]: ").strip().lower() or "moderate"
    exercises_input = input("Exercises (comma-separated): ").strip()
    exercises = [e.strip() for e in exercises_input.split(",")] if exercises_input else []
    notes = input("Notes (optional): ").strip()

    return log_workout(workout_type, duration, exercises, intensity, notes)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        command = sys.argv[1]
        if command == "cardio" and len(sys.argv) >= 3:
            duration = int(sys.argv[2])
            distance = float(sys.argv[3]) if len(sys.argv) > 3 else None
            quick_log_cardio(duration, distance)
        elif command == "strength" and len(sys.argv) >= 3:
            duration = int(sys.argv[2])
            exercises = sys.argv[3:] if len(sys.argv) > 3 else ["workout"]
            quick_log_strength(duration, exercises)
        elif command == "yoga" and len(sys.argv) >= 3:
            duration = int(sys.argv[2])
            quick_log_yoga(duration)
        else:
            print("Usage:")
            print("  python log_workout.py cardio <minutes> [distance_km]")
            print("  python log_workout.py strength <minutes> <exercise1> <exercise2> ...")
            print("  python log_workout.py yoga <minutes>")
            print("  python log_workout.py  (interactive mode)")
    else:
        # Interactive mode
        interactive_log()
