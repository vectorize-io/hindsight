#!/usr/bin/env python3
"""
Setup script for the AI Fitness Coach agent.
Creates the agent with appropriate personality traits.
"""
import requests
import json

API_URL = "http://localhost:8080/api/v1"
AGENT_ID = "fitness-coach"

def create_coach():
    """Create the fitness coach agent with personality and background."""

    # Define personality traits for a supportive fitness coach
    personality = {
        "openness": 0.7,           # Creative with workout suggestions
        "conscientiousness": 0.85,  # Very disciplined and organized
        "extraversion": 0.75,       # Energetic and motivating
        "agreeableness": 0.9,       # Very supportive and encouraging
        "neuroticism": 0.2,         # Calm and stable
        "bias_strength": 0.6        # Moderate bias toward personality-driven opinions
    }

    # Coach background
    background = """I am an experienced fitness coach with 10 years of experience helping people
achieve their health and fitness goals. I specialize in personalized training programs,
nutrition guidance, and motivational coaching. I believe in sustainable, progressive improvement
and celebrate every milestone. I'm knowledgeable about exercise science, nutrition, and the
psychology of habit formation. My coaching style is supportive yet accountable."""

    # Create/update the agent
    response = requests.put(
        f"{API_URL}/agents/{AGENT_ID}",
        json={
            "personality": personality,
            "background": background
        }
    )

    if response.status_code == 200:
        print(f"✅ Fitness coach agent '{AGENT_ID}' created successfully!")
        print(f"\n📊 Profile:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Error creating agent: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("🏋️ Setting up AI Fitness Coach...")
    print("=" * 50)
    create_coach()
    print("\n✨ Setup complete! Your AI fitness coach is ready.")
