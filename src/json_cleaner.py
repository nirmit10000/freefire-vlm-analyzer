"""
This file cleans messy AI output and extracts valid JSON.
Sometimes AI adds extra text, this removes it.
"""

import json
import re


def extract_json_from_text(text):
    """
    Find and extract just the JSON part from text.
    
    Example:
        Input:  "Here's the analysis: ```json\n{...}\n```"
        Output: "{...}"
    
    Args:
        text: Raw text from AI (might have markdown, extra words)
    
    Returns:
        str: Just the JSON part
    """
    # Remove markdown code blocks (```json and ```)
    text = text.replace('```json', '')
    text = text.replace('```', '')
    
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    # If can't find JSON brackets, return original text
    if start == -1 or end == -1 or end <= start:
        return text.strip()
    
    # Extract just the {...} part
    return text[start:end + 1].strip()


def clean_model_output(raw_output):
    """
    Main function: takes messy AI output, returns clean JSON.
    
    Args:
        raw_output: Raw text from AI model
    
    Returns:
        dict: Parsed JSON as Python dictionary
    
    Raises:
        ValueError: If can't extract valid JSON
    """
    # Step 1: Extract JSON text
    json_text = extract_json_from_text(raw_output)
    
    if not json_text:
        raise ValueError("No JSON found in model output")
    
    # Step 2: Try to parse it
    try:
        result = json.loads(json_text)
        return result
    except json.JSONDecodeError as e:
        # If parsing fails, raise error with details
        raise ValueError(f"Invalid JSON from model: {e}")


def create_error_response(error_msg):
    """
    Create a fake response when analysis fails.
    This ensures we always return something, even on error.
    
    Args:
        error_msg: What went wrong
    
    Returns:
        dict: Error response in expected format
    """
    return {
        "error": True,
        "error_message": error_msg,
        "gameplay_summary": "Analysis failed",
        "attributes": {
            "session_game_mode": "Unknown",
            "map_area_type": "Unknown",
            "player_inferred_skill_level": "Unknown",
            "weapon_type_used": "Unknown",
            "aggressive_playstyle_observed": False,
            "team_coordination_observed": False,
            "objective_pursuit_observed": False,
            "exploration_behavior_observed": False,
            "social_interaction_observed": False,
            "match_result": "Not Applicable",
            "eliminations_count": 0,
            "player_deaths_count": 0,
            "rounds_lost_prior": 0,
            "unfair_gameplay_attributed_to_hacker": False,
            "repeated_failures_observed": False,
            "monetization_elements_present_lobby": False,
            "player_idle_in_lobby": False,
            "shop_opened_observed": False,
            "purchase_event_observed": False,
            "session_ended_cleanly": True,
        },
        "major_events": []
    }