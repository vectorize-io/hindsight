#!/usr/bin/env python3
"""
Meal logging script for the AI Fitness Coach.
Allows logging meals with nutrition information.
"""
import requests
from datetime import datetime
import sys

API_URL = "http://localhost:8080/api/v1"
AGENT_ID = "fitness-coach"

def log_meal(meal_type, foods, calories=None, protein=None, carbs=None, fats=None, notes=""):
    """
    Log a meal.

    Args:
        meal_type: Type of meal ("breakfast", "lunch", "dinner", "snack")
        foods: List of foods eaten
        calories: Total calories (optional)
        protein: Protein in grams (optional)
        carbs: Carbs in grams (optional)
        fats: Fats in grams (optional)
        notes: Additional notes
    """

    # Build meal description
    foods_str = ", ".join(foods)

    content = f"User ate {foods_str} for {meal_type}. "

    # Add nutrition info if provided
    nutrition_parts = []
    if calories:
        nutrition_parts.append(f"{calories} calories")
    if protein:
        nutrition_parts.append(f"{protein}g protein")
    if carbs:
        nutrition_parts.append(f"{carbs}g carbs")
    if fats:
        nutrition_parts.append(f"{fats}g fat")

    if nutrition_parts:
        content += "Nutrition: " + ", ".join(nutrition_parts) + ". "

    if notes:
        content += f"Notes: {notes}"

    # Prepare data for API
    timestamp = datetime.now().isoformat()

    payload = {
        "agent_id": AGENT_ID,
        "items": [{
            "content": content,
            "context": f"meal-{meal_type}",
            "event_date": timestamp
        }]
    }

    # Send to API
    response = requests.post(
        f"{API_URL}/agents/{AGENT_ID}/memories",
        json=payload
    )

    if response.status_code == 200:
        print(f"✅ Meal logged successfully!")
        print(f"   Type: {meal_type}")
        print(f"   Foods: {foods_str}")
        if nutrition_parts:
            print(f"   Nutrition: {', '.join(nutrition_parts)}")
        return True
    else:
        print(f"❌ Error logging meal: {response.status_code}")
        print(response.text)
        return False

def quick_log_breakfast(foods):
    """Quick log for breakfast."""
    return log_meal("breakfast", foods)

def quick_log_lunch(foods, calories=None):
    """Quick log for lunch."""
    return log_meal("lunch", foods, calories=calories)

def quick_log_dinner(foods, calories=None, protein=None):
    """Quick log for dinner."""
    return log_meal("dinner", foods, calories=calories, protein=protein)

def interactive_log():
    """Interactive meal logging."""
    print("🍽️  Meal Logger")
    print("=" * 50)

    meal_type = input("Meal type (breakfast/lunch/dinner/snack): ").strip().lower()
    foods_input = input("Foods eaten (comma-separated): ").strip()
    foods = [f.strip() for f in foods_input.split(",")]

    # Optional nutrition info
    calories_input = input("Calories (optional, press Enter to skip): ").strip()
    calories = int(calories_input) if calories_input else None

    protein_input = input("Protein (g) (optional): ").strip()
    protein = int(protein_input) if protein_input else None

    carbs_input = input("Carbs (g) (optional): ").strip()
    carbs = int(carbs_input) if carbs_input else None

    fats_input = input("Fats (g) (optional): ").strip()
    fats = int(fats_input) if fats_input else None

    notes = input("Notes (optional): ").strip()

    return log_meal(meal_type, foods, calories, protein, carbs, fats, notes)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Command-line mode
        meal_type = sys.argv[1]
        foods = sys.argv[2:]
        if meal_type in ["breakfast", "lunch", "dinner", "snack"]:
            log_meal(meal_type, foods)
        else:
            print("Usage:")
            print("  python log_meal.py breakfast <food1> <food2> ...")
            print("  python log_meal.py lunch <food1> <food2> ...")
            print("  python log_meal.py dinner <food1> <food2> ...")
            print("  python log_meal.py  (interactive mode)")
    else:
        # Interactive mode
        interactive_log()
