#!/usr/bin/env python3
"""
AI Fitness Coach Chat Interface.
Ask questions and get personalized advice based on your workout and meal history.
"""
import requests
import json
import sys

API_URL = "http://localhost:8080/api/v1"
AGENT_ID = "fitness-coach"

def ask_coach(question, context=None, thinking_budget=100):
    """
    Ask the fitness coach a question.

    Args:
        question: Your question for the coach
        context: Optional additional context
        thinking_budget: Thinking budget for retrieval (default: 100)

    Returns:
        Coach's response with facts it was based on
    """

    payload = {
        "query": question,
        "thinking_budget": thinking_budget
    }

    if context:
        payload["context"] = context

    # Call the think API
    response = requests.post(
        f"{API_URL}/agents/{AGENT_ID}/think",
        json=payload
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def search_history(query, fact_types=None, top_k=10):
    """
    Search workout/meal history.

    Args:
        query: Search query
        fact_types: List of fact types to search (default: all)
        top_k: Number of results to return

    Returns:
        Search results
    """

    payload = {
        "agent_id": AGENT_ID,
        "query": query,
        "fact_type": fact_types or ["world", "agent", "opinion"],
        "thinking_budget": 100,
        "top_k": top_k
    }

    response = requests.post(
        f"{API_URL}/agents/{AGENT_ID}/memories/search",
        json=payload
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def print_response(result):
    """Pretty print the coach's response."""

    print("\n" + "=" * 70)
    print("🏋️ COACH'S ADVICE")
    print("=" * 70)
    print(f"\n{result['text']}\n")

    # Show what the advice was based on
    if result.get('based_on'):
        print("-" * 70)
        print("📊 BASED ON:")
        print("-" * 70)

        for fact in result['based_on']:
            fact_type = fact.get('type', 'unknown')
            emoji = {"world": "🌍", "agent": "👤", "opinion": "💭"}.get(fact_type, "📝")

            print(f"\n{emoji} [{fact_type.upper()}]")
            print(f"   {fact['text']}")

            if fact.get('event_date'):
                print(f"   📅 {fact['event_date'][:10]}")

    # Show new opinions formed
    if result.get('new_opinions'):
        print("\n" + "-" * 70)
        print("✨ NEW INSIGHTS FORMED:")
        print("-" * 70)
        for opinion in result['new_opinions']:
            confidence = opinion.get('confidence', 0)
            print(f"\n💡 {opinion['text']}")
            print(f"   Confidence: {confidence:.0%}")

    print("\n" + "=" * 70 + "\n")

def interactive_chat():
    """Interactive chat with the fitness coach."""

    print("\n🏋️ AI FITNESS COACH CHAT")
    print("=" * 70)
    print("Ask me anything about your fitness journey!")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Keep up the great work! See you next time!")
                break

            # Get coach's response
            print("\n🤔 Coach is thinking...")
            result = ask_coach(question)

            if result:
                print_response(result)

        except KeyboardInterrupt:
            print("\n\n👋 Keep up the great work! See you next time!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        print(f"\nYou: {question}")
        result = ask_coach(question)
        if result:
            print_response(result)
    else:
        # Interactive mode
        interactive_chat()

if __name__ == "__main__":
    main()
