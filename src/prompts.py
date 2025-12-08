"""
This file contains the prompt we send to the AI model.
The prompt tells the AI what to look for in Free Fire gameplay.
"""

# This is what we ask the AI to do for each frame
FREE_FIRE_PROMPT = '''Analyze this Free Fire gameplay frame and respond with valid JSON ONLY.

RULES:
1. Output must be valid JSON (starts with { ends with })
2. No markdown, no extra text, JUST JSON
3. Use exact field names below
4. If unsure, use "Unknown" for text, 0 for numbers, false for yes/no

JSON FORMAT:
{
  "gameplay_summary": "Describe what's happening in 2-3 sentences",
  "attributes": {
    "session_game_mode": "One of: Battle Royale, Clash Squad, Training Mode, Social Hub, Unknown",
    "map_area_type": "One of: Training Room, Combat Zone, Social Hub, Unknown",
    "player_inferred_skill_level": "One of: Novice, Intermediate, Expert, Unknown",
    "weapon_type_used": "One of: Sniper Rifle, Assault Rifle, Pistol, Shotgun, Melee, SMG, Other, Unknown",
    "aggressive_playstyle_observed": false,
    "team_coordination_observed": false,
    "objective_pursuit_observed": false,
    "exploration_behavior_observed": false,
    "social_interaction_observed": false,
    "match_result": "One of: Win, Loss, Draw, Not Applicable, Incomplete",
    "eliminations_count": 0,
    "player_deaths_count": 0,
    "rounds_lost_prior": 0,
    "unfair_gameplay_attributed_to_hacker": false,
    "repeated_failures_observed": false,
    "monetization_elements_present_lobby": false,
    "player_idle_in_lobby": false,
    "shop_opened_observed": false,
    "purchase_event_observed": false,
    "session_ended_cleanly": true
  },
  "major_events": [
    {
      "timestamp": "00:00",
      "event_type": "Kill, Death, Shop Interaction, etc.",
      "description": "What happened"
    }
  ]
}

WHAT TO LOOK FOR:
- Skill: Gloo Wall usage, headshots, movement
- Frustration: Repeated deaths, stuck in same spot
- Money: Shop open, purchases, vending machines
- Social: Emotes, voice chat, team play'''


def get_analysis_prompt():
    """
    Returns the prompt to send to the AI.
    
    Returns:
        str: The complete prompt text
    """
    return FREE_FIRE_PROMPT


# Simple test prompt (used by validation script)
VALIDATION_PROMPT = '''Look at this image and respond with ONLY this JSON:
{
  "test": "success",
  "image_visible": true
}'''


def get_validation_prompt():
    """
    Returns a simple test prompt.
    
    Returns:
        str: Simple prompt for testing
    """
    return VALIDATION_PROMPT