#!/usr/bin/env python3
"""
CoT Fine-tuning Script for Qwen2.5-14B-Instruct

Fine-tunes the base model with Chain-of-Thought examples showing:
1. How to use recipe_semantic_search tool
2. How to reason about macro targets
3. How to adjust quantities to meet targets (±5%)
4. How to structure the final JSON answer
5. Examples that lead to high rewards (schema valid, macros correct, variety)

PREREQUISITES:
--------------
1. Install dependencies: `uv sync`
2. Install unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
3. Set up environment variables in .env file:
   - OPENAI_API_KEY (required, for generating examples)
   - PINECONE_API_KEY (required, for recipe search)
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Fix for PyTorch 2.7.1+ compatibility
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:False')

import torch
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from tqdm import tqdm
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Ensure we're in the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Validate required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")

if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-CoT-Fitness"
SEED = 42
MAX_EXAMPLES = 1  # Generate just one valid example for verification
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 8192

# ============================================================================
# DATA MODELS
# ============================================================================

class Scenario(BaseModel):
    question: str
    split: str
    id: str
    daily_cal_target: Optional[int] = None
    daily_prot_target: Optional[int] = None
    daily_carb_target: Optional[int] = None
    daily_fat_target: Optional[int] = None
    banned_keywords: Optional[List[str]] = None

# ============================================================================
# SEED DATA (For Constructing Valid Examples)
# ============================================================================

MEAL_SEED_DATA = [
    # === BREAKFASTS ===
    {
        "name": "Greek Yogurt Bowl with Berries",
        "recipe": "greek_yogurt_bowl_berries",
        "minutes": 5,
        "description": "High-protein yogurt bowl with mixed berries, honey, and granola.",
        "ingredients": (
            "200g non-fat Greek yogurt\n"
            "50g mixed berries\n"
            "15g honey\n"
            "20g granola\n"
            "5g chia seeds"
        ),
        "steps": (
            "1. Add Greek yogurt to a bowl.\n"
            "2. Top with berries, granola, and chia seeds.\n"
            "3. Drizzle honey on top and serve."
        ),
        "calories": 350,
        "proteins": 30,
        "carbs": 40,
        "fats": 8,
        "sugars": 25,
        "tags": "breakfast,high_protein,vegetarian,quick"
    },
    {
        "name": "Oats with Whey and Banana",
        "recipe": "oats_whey_banana",
        "minutes": 10,
        "description": "Warm oats cooked with milk, whey protein, and sliced banana.",
        "ingredients": (
            "60g rolled oats\n"
            "250ml low-fat milk or water\n"
            "1 scoop whey protein (30g)\n"
            "1 medium banana, sliced\n"
            "5g peanut butter (optional)"
        ),
        "steps": (
            "1. Cook oats with milk or water until soft.\n"
            "2. Remove from heat and stir in whey protein.\n"
            "3. Top with sliced banana and peanut butter."
        ),
        "calories": 420,
        "proteins": 28,
        "carbs": 55,
        "fats": 10,
        "sugars": 18,
        "tags": "breakfast,high_protein,muscle_gain,vegetarian"
    },
    {
        "name": "Avocado Toast with Smoked Salmon",
        "recipe": "avocado_toast_salmon",
        "minutes": 8,
        "description": "Whole-grain toast topped with mashed avocado and smoked salmon.",
        "ingredients": (
            "2 slices whole-grain bread\n"
            "60g avocado\n"
            "60g smoked salmon\n"
            "Lemon juice, salt, pepper"
        ),
        "steps": (
            "1. Toast the bread.\n"
            "2. Mash avocado with lemon juice, salt, and pepper.\n"
            "3. Spread on toast and top with smoked salmon."
        ),
        "calories": 430,
        "proteins": 24,
        "carbs": 35,
        "fats": 22,
        "sugars": 3,
        "tags": "breakfast,lunch,pescatarian,healthy_fats,moderate_carb"
    },
    {
        "name": "Tofu Scramble with Veggies",
        "recipe": "tofu_scramble_veggies",
        "minutes": 15,
        "description": "Vegan tofu scramble with mixed vegetables and spices.",
        "ingredients": (
            "150g firm tofu\n"
            "50g spinach\n"
            "50g bell pepper\n"
            "30g onion\n"
            "1 tsp olive oil\n"
            "Turmeric, salt, pepper"
        ),
        "steps": (
            "1. Crumble tofu into a pan with heated olive oil.\n"
            "2. Add chopped onion and bell pepper, cook until soft.\n"
            "3. Add spinach and spices, cook until wilted."
        ),
        "calories": 260,
        "proteins": 20,
        "carbs": 12,
        "fats": 14,
        "sugars": 4,
        "tags": "breakfast,vegan,low_carb,high_protein"
    },
    {
        "name": "Protein Berry Smoothie",
        "recipe": "protein_berry_smoothie",
        "minutes": 5,
        "description": "Quick smoothie with protein powder, berries, oats, and milk.",
        "ingredients": (
            "1 scoop protein powder (30g)\n"
            "100g mixed berries\n"
            "30g oats\n"
            "200ml low-fat milk or plant milk\n"
            "Ice cubes (optional)"
        ),
        "steps": (
            "1. Add all ingredients to a blender.\n"
            "2. Blend until smooth and serve immediately."
        ),
        "calories": 350,
        "proteins": 28,
        "carbs": 40,
        "fats": 7,
        "sugars": 18,
        "tags": "breakfast,snack,high_protein,quick,drink"
    },

    # === LUNCH / DINNER ===
    {
        "name": "Grilled Chicken with Rice and Broccoli",
        "recipe": "chicken_rice_broccoli",
        "minutes": 30,
        "description": "Classic bodybuilding meal with grilled chicken breast, jasmine rice, and steamed broccoli.",
        "ingredients": (
            "150g chicken breast\n"
            "150g cooked jasmine rice\n"
            "100g broccoli\n"
            "1 tsp olive oil\n"
            "Salt, pepper, herbs"
        ),
        "steps": (
            "1. Season chicken and grill until cooked through.\n"
            "2. Cook rice according to package instructions.\n"
            "3. Steam broccoli and drizzle with olive oil.\n"
            "4. Serve chicken with rice and broccoli."
        ),
        "calories": 520,
        "proteins": 45,
        "carbs": 55,
        "fats": 12,
        "sugars": 3,
        "tags": "lunch,dinner,high_protein,muscle_gain,balanced"
    },
    {
        "name": "Salmon with Quinoa and Asparagus",
        "recipe": "salmon_quinoa_asparagus",
        "minutes": 25,
        "description": "Oven-baked salmon served with quinoa and roasted asparagus.",
        "ingredients": (
            "150g salmon fillet\n"
            "150g cooked quinoa\n"
            "100g asparagus\n"
            "1 tsp olive oil\n"
            "Lemon, salt, pepper"
        ),
        "steps": (
            "1. Season salmon with lemon, salt, and pepper and bake at 180°C for ~15 minutes.\n"
            "2. Cook quinoa according to package instructions.\n"
            "3. Toss asparagus with olive oil, salt, and roast until tender.\n"
            "4. Plate salmon with quinoa and asparagus."
        ),
        "calories": 538,
        "proteins": 40,
        "carbs": 45,
        "fats": 22,
        "sugars": 3,
        "tags": "lunch,dinner,pescatarian,high_protein,healthy_fats,gluten_free"
    },
    {
        "name": "Turkey Bolognese with Whole-Wheat Pasta",
        "recipe": "turkey_bolognese_pasta",
        "minutes": 35,
        "description": "Lean turkey mince cooked in tomato sauce over whole-wheat pasta.",
        "ingredients": (
            "120g whole-wheat pasta (dry)\n"
            "150g lean turkey mince\n"
            "100g tomato passata\n"
            "30g onion, 1 garlic clove\n"
            "1 tsp olive oil, Italian herbs"
        ),
        "steps": (
            "1. Cook pasta according to package instructions.\n"
            "2. Sauté onion and garlic in olive oil, add turkey mince.\n"
            "3. Add tomato passata and herbs, simmer 10–15 minutes.\n"
            "4. Serve sauce over pasta."
        ),
        "calories": 566,
        "proteins": 40,
        "carbs": 70,
        "fats": 14,
        "sugars": 9,
        "tags": "lunch,dinner,high_protein,muscle_gain"
    },
    {
        "name": "Tofu Veggie Stir-Fry with Brown Rice",
        "recipe": "tofu_stirfry_brown_rice",
        "minutes": 25,
        "description": "Stir-fried tofu with mixed vegetables and soy sauce over brown rice.",
        "ingredients": (
            "150g firm tofu\n"
            "150g cooked brown rice\n"
            "100g mixed vegetables (broccoli, carrots, peppers)\n"
            "1 tbsp soy sauce (low sodium)\n"
            "1 tsp sesame oil"
        ),
        "steps": (
            "1. Stir-fry tofu cubes in sesame oil until golden.\n"
            "2. Add vegetables and cook until tender-crisp.\n"
            "3. Add soy sauce and toss.\n"
            "4. Serve over brown rice."
        ),
        "calories": 520,
        "proteins": 26,
        "carbs": 60,
        "fats": 18,
        "sugars": 7,
        "tags": "lunch,dinner,vegan,high_fiber,balanced"
    },
    {
        "name": "Chickpea Buddha Bowl",
        "recipe": "chickpea_buddha_bowl",
        "minutes": 20,
        "description": "Bowl with chickpeas, quinoa, roasted veggies, and tahini dressing.",
        "ingredients": (
            "120g cooked chickpeas\n"
            "100g cooked quinoa\n"
            "80g roasted sweet potato\n"
            "50g mixed greens\n"
            "1 tbsp tahini, lemon juice"
        ),
        "steps": (
            "1. Roast sweet potato cubes until tender.\n"
            "2. Assemble bowl with quinoa, chickpeas, sweet potato, and greens.\n"
            "3. Drizzle tahini mixed with lemon juice and water."
        ),
        "calories": 540,
        "proteins": 22,
        "carbs": 75,
        "fats": 16,
        "sugars": 10,
        "tags": "lunch,dinner,vegan,high_fiber,vegetarian"
    },
    {
        "name": "Chicken Salad Bowl (Low-Carb)",
        "recipe": "chicken_salad_lowcarb",
        "minutes": 15,
        "description": "Low-carb salad with grilled chicken, mixed greens, tomato, cucumber, and olive oil.",
        "ingredients": (
            "150g grilled chicken breast\n"
            "80g mixed salad greens\n"
            "50g cucumber\n"
            "50g tomato\n"
            "10g olive oil\n"
            "Vinegar, salt, pepper"
        ),
        "steps": (
            "1. Slice grilled chicken.\n"
            "2. Add chopped vegetables and greens to a bowl.\n"
            "3. Top with chicken and drizzle olive oil and vinegar."
        ),
        "calories": 350,
        "proteins": 40,
        "carbs": 8,
        "fats": 16,
        "sugars": 5,
        "tags": "lunch,dinner,low_carb,high_protein,fat_loss"
    },
    {
        "name": "Shrimp Zoodle Stir-Fry",
        "recipe": "shrimp_zoodle_stirfry",
        "minutes": 20,
        "description": "Shrimp stir-fried with zucchini noodles and vegetables.",
        "ingredients": (
            "150g shrimp\n"
            "150g zucchini noodles\n"
            "50g bell pepper\n"
            "1 tsp olive oil\n"
            "Garlic, soy sauce"
        ),
        "steps": (
            "1. Sauté garlic in olive oil.\n"
            "2. Add shrimp and cook until pink.\n"
            "3. Add zucchini noodles and peppers, stir-fry briefly.\n"
            "4. Add a splash of soy sauce and serve."
        ),
        "calories": 276,
        "proteins": 32,
        "carbs": 10,
        "fats": 12,
        "sugars": 5,
        "tags": "lunch,dinner,pescatarian,low_carb,high_protein"
    },

    # === SNACKS / LIGHT MEALS ===
    {
        "name": "Cottage Cheese with Pineapple",
        "recipe": "cottage_cheese_pineapple",
        "minutes": 5,
        "description": "High-protein cottage cheese with pineapple chunks.",
        "ingredients": (
            "200g low-fat cottage cheese\n"
            "80g pineapple chunks (fresh or canned in juice)"
        ),
        "steps": (
            "1. Add cottage cheese to a bowl.\n"
            "2. Top with pineapple and serve."
        ),
        "calories": 230,
        "proteins": 26,
        "carbs": 22,
        "fats": 3,
        "sugars": 18,
        "tags": "snack,high_protein,vegetarian,quick"
    },
    {
        "name": "Apple with Peanut Butter",
        "recipe": "apple_peanut_butter",
        "minutes": 3,
        "description": "Simple snack of sliced apple with peanut butter.",
        "ingredients": (
            "1 medium apple\n"
            "20g peanut butter"
        ),
        "steps": (
            "1. Slice apple.\n"
            "2. Serve with peanut butter for dipping."
        ),
        "calories": 200,
        "proteins": 4,
        "carbs": 24,
        "fats": 10,
        "sugars": 17,
        "tags": "snack,vegetarian,healthy_fats,quick"
    },
    {
        "name": "Mixed Nuts Portion",
        "recipe": "mixed_nuts_portion",
        "minutes": 1,
        "description": "A small handful of mixed nuts.",
        "ingredients": "30g mixed nuts (almonds, walnuts, cashews)",
        "steps": "1. Portion out 30g of mixed nuts and eat.",
        "calories": 180,
        "proteins": 6,
        "carbs": 6,
        "fats": 15,
        "sugars": 2,
        "tags": "snack,vegan,healthy_fats,low_carb"
    },
    {
        "name": "Hummus with Carrot Sticks",
        "recipe": "hummus_carrot_sticks",
        "minutes": 5,
        "description": "Carrot sticks dipped in classic hummus.",
        "ingredients": (
            "50g hummus\n"
            "100g carrot sticks"
        ),
        "steps": (
            "1. Slice carrots into sticks.\n"
            "2. Serve with hummus for dipping."
        ),
        "calories": 170,
        "proteins": 6,
        "carbs": 20,
        "fats": 7,
        "sugars": 7,
        "tags": "snack,vegan,vegetarian,high_fiber"
    },
    {
        "name": "Rice Cakes with Peanut Butter",
        "recipe": "rice_cakes_peanut_butter",
        "minutes": 3,
        "description": "Rice cakes spread with peanut butter.",
        "ingredients": (
            "2 plain rice cakes\n"
            "20g peanut butter"
        ),
        "steps": (
            "1. Spread peanut butter on rice cakes.\n"
            "2. Serve immediately."
        ),
        "calories": 190,
        "proteins": 6,
        "carbs": 22,
        "fats": 9,
        "sugars": 2,
        "tags": "snack,vegetarian,quick"
    },
    {
        "name": "Protein Bar (Generic)",
        "recipe": "protein_bar_generic",
        "minutes": 1,
        "description": "Store-bought protein bar (generic macro profile).",
        "ingredients": "1 protein bar (~60g)",
        "steps": "1. Unwrap and eat.",
        "calories": 220,
        "proteins": 20,
        "carbs": 24,
        "fats": 7,
        "sugars": 10,
        "tags": "snack,high_protein,quick,convenience"
    },
]

MORE_MEAL_SEED_DATA = [
    # === BREAKFASTS ===
    {
        "name": "Egg White Omelette with Spinach and Feta",
        "recipe": "eggwhite_omelette_spinach_feta",
        "minutes": 12,
        "description": "Fluffy egg white omelette with spinach and a bit of feta cheese.",
        "ingredients": (
            "4 egg whites\n"
            "30g feta cheese\n"
            "40g spinach\n"
            "5g olive oil\n"
            "Salt, pepper"
        ),
        "steps": (
            "1. Whisk egg whites with salt and pepper.\n"
            "2. Sauté spinach in olive oil until wilted.\n"
            "3. Add egg whites to the pan and cook until almost set.\n"
            "4. Sprinkle feta, fold omelette, and serve."
        ),
        "calories": 220,
        "proteins": 25,
        "carbs": 3,
        "fats": 10,
        "sugars": 2,
        "tags": "breakfast,high_protein,low_carb,vegetarian"
    },
    {
        "name": "Chia Pudding with Almond Milk",
        "recipe": "chia_pudding_almond_milk",
        "minutes": 5,
        "description": "Overnight chia pudding made with almond milk and topped with fruit.",
        "ingredients": (
            "30g chia seeds\n"
            "200ml unsweetened almond milk\n"
            "10g honey or maple syrup\n"
            "30g berries"
        ),
        "steps": (
            "1. Mix chia seeds, almond milk, and sweetener in a jar.\n"
            "2. Refrigerate at least 4 hours or overnight.\n"
            "3. Top with berries before serving."
        ),
        "calories": 260,
        "proteins": 8,
        "carbs": 24,
        "fats": 14,
        "sugars": 11,
        "tags": "breakfast,vegan,high_fiber,prep_ahead"
    },
    {
        "name": "Protein Banana Pancakes",
        "recipe": "protein_banana_pancakes",
        "minutes": 15,
        "description": "High-protein pancakes made with oats, banana, and whey.",
        "ingredients": (
            "1 medium banana\n"
            "40g oats\n"
            "1 scoop whey protein (30g)\n"
            "1 egg white\n"
            "50ml milk or water"
        ),
        "steps": (
            "1. Blend all ingredients until smooth.\n"
            "2. Pour batter onto a hot non-stick pan.\n"
            "3. Cook 2–3 minutes per side until golden.\n"
            "4. Serve with optional berries or sugar-free syrup."
        ),
        "calories": 380,
        "proteins": 27,
        "carbs": 48,
        "fats": 7,
        "sugars": 16,
        "tags": "breakfast,high_protein,muscle_gain,vegetarian"
    },
    {
        "name": "Breakfast Burrito with Eggs and Beans",
        "recipe": "breakfast_burrito_eggs_beans",
        "minutes": 20,
        "description": "Hearty breakfast wrap with scrambled eggs, beans, and veggies.",
        "ingredients": (
            "1 large whole-wheat tortilla\n"
            "2 eggs\n"
            "40g black beans\n"
            "30g grated cheese\n"
            "30g bell pepper\n"
            "Salsa, salt, pepper"
        ),
        "steps": (
            "1. Scramble eggs with diced bell pepper.\n"
            "2. Warm tortilla and add eggs, beans, cheese, and salsa.\n"
            "3. Roll into a burrito and serve."
        ),
        "calories": 480,
        "proteins": 26,
        "carbs": 45,
        "fats": 20,
        "sugars": 5,
        "tags": "breakfast,lunch,high_protein,balanced"
    },

    # === LUNCH / DINNER (MEAT / FISH) ===
    {
        "name": "Beef Stir-Fry with Vegetables and Rice",
        "recipe": "beef_stirfry_veg_rice",
        "minutes": 25,
        "description": "Lean beef strips stir-fried with mixed vegetables and served with rice.",
        "ingredients": (
            "120g lean beef strips\n"
            "150g cooked white or brown rice\n"
            "100g mixed vegetables (broccoli, peppers, carrots)\n"
            "1 tbsp soy sauce\n"
            "1 tsp sesame oil"
        ),
        "steps": (
            "1. Stir-fry beef in a hot pan until browned.\n"
            "2. Add vegetables and cook until tender-crisp.\n"
            "3. Add soy sauce and sesame oil, toss well.\n"
            "4. Serve over rice."
        ),
        "calories": 580,
        "proteins": 36,
        "carbs": 65,
        "fats": 18,
        "sugars": 7,
        "tags": "lunch,dinner,high_protein,muscle_gain"
    },
    {
        "name": "Turkey Wrap with Veggies",
        "recipe": "turkey_wrap_veggies",
        "minutes": 10,
        "description": "Whole-wheat wrap filled with lean turkey, veggies, and yogurt sauce.",
        "ingredients": (
            "1 whole-wheat tortilla\n"
            "80g sliced turkey breast\n"
            "30g lettuce\n"
            "30g tomato\n"
            "20g cucumber\n"
            "20g Greek yogurt, herbs"
        ),
        "steps": (
            "1. Spread yogurt on tortilla.\n"
            "2. Add turkey slices and chopped veggies.\n"
            "3. Roll tightly into a wrap and slice in half."
        ),
        "calories": 320,
        "proteins": 26,
        "carbs": 32,
        "fats": 9,
        "sugars": 4,
        "tags": "lunch,high_protein,quick,fat_loss"
    },
    {
        "name": "Chicken Fajita Bowl",
        "recipe": "chicken_fajita_bowl",
        "minutes": 25,
        "description": "Fajita-style chicken with peppers, onions, and rice in a bowl.",
        "ingredients": (
            "150g chicken breast\n"
            "150g cooked rice\n"
            "40g bell pepper\n"
            "30g onion\n"
            "1 tsp olive oil\n"
            "Fajita seasoning"
        ),
        "steps": (
            "1. Slice chicken, peppers, and onion.\n"
            "2. Stir-fry in olive oil with fajita seasoning.\n"
            "3. Serve over rice in a bowl."
        ),
        "calories": 520,
        "proteins": 40,
        "carbs": 55,
        "fats": 14,
        "sugars": 6,
        "tags": "lunch,dinner,high_protein,balanced"
    },
    {
        "name": "Baked Cod with Potatoes and Green Beans",
        "recipe": "baked_cod_potatoes_greenbeans",
        "minutes": 30,
        "description": "Light baked cod served with potatoes and steamed green beans.",
        "ingredients": (
            "150g cod fillet\n"
            "150g potatoes\n"
            "80g green beans\n"
            "1 tsp olive oil\n"
            "Lemon, garlic, salt, pepper"
        ),
        "steps": (
            "1. Season cod with lemon, garlic, salt, and pepper.\n"
            "2. Bake at 180°C for 15–20 minutes.\n"
            "3. Boil or steam potatoes and green beans.\n"
            "4. Serve cod with potatoes and beans."
        ),
        "calories": 430,
        "proteins": 35,
        "carbs": 45,
        "fats": 10,
        "sugars": 4,
        "tags": "lunch,dinner,pescatarian,high_protein,fat_loss"
    },
    {
        "name": "Lean Beef Burger with Sweet Potato Fries",
        "recipe": "beef_burger_sweetpotato_fries",
        "minutes": 30,
        "description": "Homemade lean beef burger with baked sweet potato fries.",
        "ingredients": (
            "130g lean minced beef\n"
            "1 whole-wheat burger bun\n"
            "80g sweet potato\n"
            "Lettuce, tomato, onion\n"
            "Salt, pepper, spices"
        ),
        "steps": (
            "1. Form beef patty, season, and grill or pan fry.\n"
            "2. Cut sweet potato into fries and bake until crispy.\n"
            "3. Assemble burger with salad veggies.\n"
            "4. Serve with sweet potato fries."
        ),
        "calories": 600,
        "proteins": 35,
        "carbs": 70,
        "fats": 20,
        "sugars": 10,
        "tags": "lunch,dinner,muscle_gain,high_protein"
    },
    {
        "name": "Tuna Salad with Olive Oil and Beans",
        "recipe": "tuna_salad_beans",
        "minutes": 10,
        "description": "Quick tuna salad with beans, veggies, and olive oil dressing.",
        "ingredients": (
            "1 can tuna in water (~120g drained)\n"
            "80g white beans or cannellini beans\n"
            "30g red onion\n"
            "40g tomato\n"
            "10g olive oil, lemon juice"
        ),
        "steps": (
            "1. Drain tuna and beans.\n"
            "2. Chop onion and tomato.\n"
            "3. Mix everything with olive oil and lemon juice.\n"
            "4. Season and serve."
        ),
        "calories": 352,
        "proteins": 34,
        "carbs": 18,
        "fats": 16,
        "sugars": 4,
        "tags": "lunch,dinner,pescatarian,high_protein,low_carb"
    },
    {
        "name": "Sushi Bowl with Salmon and Rice",
        "recipe": "sushi_bowl_salmon_rice",
        "minutes": 25,
        "description": "Deconstructed sushi bowl with salmon, rice, and veggies.",
        "ingredients": (
            "120g salmon (raw or cooked, as preferred)\n"
            "150g cooked sushi rice\n"
            "40g cucumber\n"
            "30g carrot\n"
            "Soy sauce, rice vinegar, sesame seeds"
        ),
        "steps": (
            "1. Cook sushi rice and season with rice vinegar.\n"
            "2. Slice cucumber and carrot into matchsticks.\n"
            "3. Add rice to a bowl, top with salmon and veggies.\n"
            "4. Drizzle soy sauce and sprinkle sesame seeds."
        ),
        "calories": 560,
        "proteins": 32,
        "carbs": 65,
        "fats": 18,
        "sugars": 7,
        "tags": "lunch,dinner,pescatarian,muscle_gain"
    },

    # === LUNCH / DINNER (VEGETARIAN / VEGAN) ===
    {
        "name": "Lentil Soup with Whole-Grain Bread",
        "recipe": "lentil_soup_wholegrain_bread",
        "minutes": 35,
        "description": "Hearty lentil soup served with a slice of whole-grain bread.",
        "ingredients": (
            "80g dry lentils\n"
            "40g carrot\n"
            "30g onion\n"
            "1 garlic clove\n"
            "1 tsp olive oil\n"
            "250ml vegetable broth\n"
            "1 slice whole-grain bread"
        ),
        "steps": (
            "1. Sauté onion, garlic, and carrot in olive oil.\n"
            "2. Add lentils and vegetable broth.\n"
            "3. Simmer 20–25 minutes until lentils are tender.\n"
            "4. Serve with bread."
        ),
        "calories": 430,
        "proteins": 22,
        "carbs": 67,
        "fats": 8,
        "sugars": 7,
        "tags": "lunch,dinner,vegan,high_fiber,vegetarian"
    },
    {
        "name": "Vegetable Chickpea Curry with Rice",
        "recipe": "veggie_chickpea_curry_rice",
        "minutes": 30,
        "description": "Mild chickpea and vegetable curry served over rice.",
        "ingredients": (
            "120g cooked chickpeas\n"
            "100g mixed vegetables (cauliflower, peas, carrots)\n"
            "100g tomato passata or chopped tomatoes\n"
            "150g cooked rice\n"
            "Curry powder, garlic, onion"
        ),
        "steps": (
            "1. Sauté onion and garlic.\n"
            "2. Add vegetables and chickpeas, cook a few minutes.\n"
            "3. Add tomatoes and curry powder, simmer 15–20 minutes.\n"
            "4. Serve over rice."
        ),
        "calories": 500,
        "proteins": 19,
        "carbs": 88,
        "fats": 8,
        "sugars": 12,
        "tags": "lunch,dinner,vegan,high_fiber,balanced"
    },
    {
        "name": "Grilled Halloumi Salad Bowl",
        "recipe": "grilled_halloumi_salad_bowl",
        "minutes": 15,
        "description": "Mediterranean salad with grilled halloumi cheese and veggies.",
        "ingredients": (
            "70g halloumi cheese\n"
            "80g mixed salad greens\n"
            "40g tomato\n"
            "40g cucumber\n"
            "10g olive oil\n"
            "Lemon juice, herbs"
        ),
        "steps": (
            "1. Grill halloumi slices until golden.\n"
            "2. Add salad greens and chopped veggies to a bowl.\n"
            "3. Top with halloumi and drizzle olive oil and lemon juice."
        ),
        "calories": 360,
        "proteins": 18,
        "carbs": 9,
        "fats": 28,
        "sugars": 5,
        "tags": "lunch,dinner,vegetarian,low_carb"
    },
    {
        "name": "Eggplant and Chickpea Stew",
        "recipe": "eggplant_chickpea_stew",
        "minutes": 35,
        "description": "Slow-simmered eggplant and chickpea stew with tomatoes.",
        "ingredients": (
            "120g cooked chickpeas\n"
            "100g eggplant\n"
            "80g tomato passata\n"
            "30g onion\n"
            "1 tsp olive oil\n"
            "Garlic, spices"
        ),
        "steps": (
            "1. Sauté onion and garlic in olive oil.\n"
            "2. Add chopped eggplant and cook until softened.\n"
            "3. Add chickpeas and tomato passata, simmer 20 minutes.\n"
            "4. Serve as a main or with rice/bread."
        ),
        "calories": 320,
        "proteins": 13,
        "carbs": 42,
        "fats": 9,
        "sugars": 11,
        "tags": "lunch,dinner,vegan,high_fiber,vegetarian"
    },
    {
        "name": "Vegan Lentil Bolognese with Pasta",
        "recipe": "vegan_lentil_bolognese_pasta",
        "minutes": 35,
        "description": "Pasta with a rich tomato and lentil Bolognese sauce.",
        "ingredients": (
            "120g whole-wheat pasta (dry)\n"
            "80g dry red lentils\n"
            "100g tomato passata\n"
            "30g onion, garlic\n"
            "Italian herbs"
        ),
        "steps": (
            "1. Cook pasta according to package instructions.\n"
            "2. Simmer lentils with onion, garlic, and tomato passata until tender.\n"
            "3. Season with herbs and serve sauce over pasta."
        ),
        "calories": 573,
        "proteins": 27,
        "carbs": 105,
        "fats": 5,
        "sugars": 12,
        "tags": "lunch,dinner,vegan,high_fiber"
    },

    # === SNACKS / LIGHT PROTEIN OPTIONS ===
    {
        "name": "Greek Yogurt with Almonds and Honey",
        "recipe": "greek_yogurt_almonds_honey",
        "minutes": 5,
        "description": "Simple snack of Greek yogurt topped with almonds and honey.",
        "ingredients": (
            "170g non-fat Greek yogurt\n"
            "15g almonds\n"
            "10g honey"
        ),
        "steps": (
            "1. Add yogurt to a bowl.\n"
            "2. Top with almonds and drizzle honey."
        ),
        "calories": 260,
        "proteins": 20,
        "carbs": 22,
        "fats": 10,
        "sugars": 17,
        "tags": "snack,high_protein,vegetarian"
    },
    {
        "name": "Boiled Eggs and Veggie Sticks",
        "recipe": "boiled_eggs_veggie_sticks",
        "minutes": 12,
        "description": "Hard-boiled eggs served with carrot and cucumber sticks.",
        "ingredients": (
            "2 eggs\n"
            "60g carrot sticks\n"
            "60g cucumber sticks"
        ),
        "steps": (
            "1. Boil eggs for 8–10 minutes, cool and peel.\n"
            "2. Slice veggies into sticks.\n"
            "3. Serve together as a snack."
        ),
        "calories": 220,
        "proteins": 14,
        "carbs": 8,
        "fats": 14,
        "sugars": 5,
        "tags": "snack,high_protein,low_carb"
    },
    {
        "name": "Edamame with Sea Salt",
        "recipe": "edamame_sea_salt",
        "minutes": 8,
        "description": "Boiled edamame sprinkled with sea salt.",
        "ingredients": "100g edamame in pods\nSalt",
        "steps": (
            "1. Boil edamame for 4–5 minutes.\n"
            "2. Drain and sprinkle with salt.\n"
            "3. Squeeze beans from pods to eat."
        ),
        "calories": 140,
        "proteins": 12,
        "carbs": 12,
        "fats": 5,
        "sugars": 2,
        "tags": "snack,vegan,high_protein,high_fiber"
    },
    {
        "name": "Dark Chocolate with Almonds",
        "recipe": "dark_chocolate_almonds",
        "minutes": 1,
        "description": "Small portion of dark chocolate with almonds.",
        "ingredients": (
            "15g dark chocolate (70%+)\n"
            "15g almonds"
        ),
        "steps": "1. Portion out chocolate and almonds and enjoy.",
        "calories": 190,
        "proteins": 4,
        "carbs": 10,
        "fats": 15,
        "sugars": 6,
        "tags": "snack,vegetarian,healthy_fats"
    },
    {
        "name": "Casein Protein Shake",
        "recipe": "casein_protein_shake",
        "minutes": 2,
        "description": "Slow-digesting casein shake, ideal before bed.",
        "ingredients": (
            "1 scoop casein protein (30g)\n"
            "250ml water or milk"
        ),
        "steps": (
            "1. Add casein powder and liquid to a shaker.\n"
            "2. Shake well and drink."
        ),
        "calories": 130,
        "proteins": 24,
        "carbs": 4,
        "fats": 2,
        "sugars": 2,
        "tags": "snack,high_protein,drink,night_snack"
    },
    {
        "name": "Air-Popped Popcorn (Light)",
        "recipe": "airpopped_popcorn_light",
        "minutes": 5,
        "description": "Low-calorie air-popped popcorn snack.",
        "ingredients": "20g popcorn kernels\nSalt or herbs",
        "steps": (
            "1. Air-pop popcorn kernels.\n"
            "2. Season lightly with salt or herbs."
        ),
        "calories": 90,
        "proteins": 3,
        "carbs": 18,
        "fats": 1,
        "sugars": 0,
        "tags": "snack,vegan,low_fat,high_fiber"
    },
]

EVEN_MORE_MEAL_SEED_DATA = [
    # === BREAKFASTS (10) ===
    {
        "name": "High-Fiber Berry Overnight Oats",
        "recipe": "high_fiber_berry_overnight_oats",
        "minutes": 5,
        "description": "Overnight oats with berries and chia seeds.",
        "ingredients": "rolled oats, milk, chia seeds, mixed berries, honey",
        "steps": "Mix everything in a jar, refrigerate overnight, top with berries before serving.",
        "calories": 380,
        "proteins": 16,
        "carbs": 55,
        "fats": 10,
        "sugars": 18,
        "tags": "breakfast,high_fiber,prep_ahead,vegetarian"
    },
    {
        "name": "Spinach Mushroom Omelette",
        "recipe": "spinach_mushroom_omelette",
        "minutes": 12,
        "description": "Egg omelette with spinach and mushrooms.",
        "ingredients": "eggs, spinach, mushrooms, olive oil, salt, pepper",
        "steps": "Sauté mushrooms and spinach, add beaten eggs, cook until set, fold and serve.",
        "calories": 300,
        "proteins": 20,
        "carbs": 6,
        "fats": 21,
        "sugars": 3,
        "tags": "breakfast,high_protein,low_carb"
    },
    {
        "name": "Cottage Cheese Berry Toast",
        "recipe": "cottage_cheese_berry_toast",
        "minutes": 7,
        "description": "Whole-grain toast topped with cottage cheese and berries.",
        "ingredients": "whole-grain bread, cottage cheese, mixed berries, honey",
        "steps": "Toast bread, spread cottage cheese, top with berries and honey.",
        "calories": 280,
        "proteins": 18,
        "carbs": 32,
        "fats": 7,
        "sugars": 12,
        "tags": "breakfast,snack,high_protein,vegetarian"
    },
    {
        "name": "Savory Breakfast Rice Bowl",
        "recipe": "savory_breakfast_rice_bowl",
        "minutes": 15,
        "description": "Rice bowl with egg, veggies, and soy sauce.",
        "ingredients": "cooked rice, egg, spinach, spring onion, soy sauce, sesame oil",
        "steps": "Fry egg, sauté veggies, add rice and soy sauce, top with egg.",
        "calories": 420,
        "proteins": 18,
        "carbs": 55,
        "fats": 14,
        "sugars": 4,
        "tags": "breakfast,lunch,balanced"
    },
    {
        "name": "Peanut Butter Banana Smoothie",
        "recipe": "peanut_butter_banana_smoothie",
        "minutes": 5,
        "description": "Creamy smoothie with banana, peanut butter, and milk.",
        "ingredients": "banana, peanut butter, milk, oats, ice",
        "steps": "Blend all ingredients until smooth and serve.",
        "calories": 430,
        "proteins": 16,
        "carbs": 52,
        "fats": 17,
        "sugars": 24,
        "tags": "breakfast,snack,drink,energy"
    },
    {
        "name": "Protein Muesli Bowl",
        "recipe": "protein_muesli_bowl",
        "minutes": 5,
        "description": "Muesli with milk and added protein powder.",
        "ingredients": "muesli, milk, protein powder, berries",
        "steps": "Stir protein powder into milk, pour over muesli, top with berries.",
        "calories": 390,
        "proteins": 28,
        "carbs": 50,
        "fats": 8,
        "sugars": 18,
        "tags": "breakfast,high_protein,muscle_gain"
    },
    {
        "name": "Smoked Salmon Breakfast Wrap",
        "recipe": "smoked_salmon_breakfast_wrap",
        "minutes": 10,
        "description": "Wrap with smoked salmon, cream cheese, and veggies.",
        "ingredients": "whole-wheat tortilla, smoked salmon, cream cheese, lettuce, cucumber",
        "steps": "Spread cream cheese, add salmon and veggies, roll into a wrap.",
        "calories": 360,
        "proteins": 22,
        "carbs": 32,
        "fats": 15,
        "sugars": 4,
        "tags": "breakfast,lunch,pescatarian"
    },
    {
        "name": "Veggie Breakfast Quesadilla",
        "recipe": "veggie_breakfast_quesadilla",
        "minutes": 15,
        "description": "Quesadilla with egg, cheese, and veggies.",
        "ingredients": "tortilla, egg, grated cheese, bell pepper, onion",
        "steps": "Cook scrambled egg with veggies, fill tortilla with mixture and cheese, toast both sides.",
        "calories": 420,
        "proteins": 22,
        "carbs": 40,
        "fats": 19,
        "sugars": 4,
        "tags": "breakfast,lunch,balanced"
    },
    {
        "name": "Apple Cinnamon Protein Oats",
        "recipe": "apple_cinnamon_protein_oats",
        "minutes": 10,
        "description": "Warm oats with apple, cinnamon, and protein powder.",
        "ingredients": "rolled oats, milk or water, protein powder, apple, cinnamon",
        "steps": "Cook oats, stir in protein powder, top with diced apple and cinnamon.",
        "calories": 410,
        "proteins": 26,
        "carbs": 55,
        "fats": 9,
        "sugars": 20,
        "tags": "breakfast,high_protein,vegetarian"
    },
    {
        "name": "Savory Greek Yogurt Bowl",
        "recipe": "savory_greek_yogurt_bowl",
        "minutes": 5,
        "description": "Greek yogurt with cucumber, olive oil, and herbs.",
        "ingredients": "Greek yogurt, cucumber, olive oil, garlic, herbs, salt",
        "steps": "Mix chopped cucumber and seasonings into yogurt, drizzle olive oil.",
        "calories": 250,
        "proteins": 18,
        "carbs": 8,
        "fats": 16,
        "sugars": 6,
        "tags": "breakfast,snack,low_carb,vegetarian"
    },

    # === LUNCH / DINNER MEAT & FISH (15) ===
    {
        "name": "High-Protein Chicken Pita",
        "recipe": "high_protein_chicken_pita",
        "minutes": 15,
        "description": "Chicken pita with salad and yogurt sauce.",
        "ingredients": "grilled chicken, whole-wheat pita, lettuce, tomato, yogurt sauce",
        "steps": "Fill pita with chicken and salad, add yogurt sauce, and serve.",
        "calories": 450,
        "proteins": 40,
        "carbs": 45,
        "fats": 12,
        "sugars": 6,
        "tags": "lunch,dinner,high_protein,fat_loss"
    },
    {
        "name": "Teriyaki Chicken Rice Bowl",
        "recipe": "teriyaki_chicken_rice_bowl",
        "minutes": 25,
        "description": "Teriyaki chicken served over rice with veggies.",
        "ingredients": "chicken breast, teriyaki sauce, rice, broccoli, carrots",
        "steps": "Cook chicken with sauce, steam veggies, serve over rice.",
        "calories": 540,
        "proteins": 38,
        "carbs": 65,
        "fats": 12,
        "sugars": 16,
        "tags": "lunch,dinner,muscle_gain,high_protein"
    },
    {
        "name": "Lemon Herb Chicken with Couscous",
        "recipe": "lemon_herb_chicken_couscous",
        "minutes": 25,
        "description": "Grilled lemon herb chicken with couscous.",
        "ingredients": "chicken breast, couscous, lemon, olive oil, herbs, vegetables",
        "steps": "Marinate chicken, grill, cook couscous, serve with chopped veggies.",
        "calories": 520,
        "proteins": 40,
        "carbs": 55,
        "fats": 14,
        "sugars": 4,
        "tags": "lunch,dinner,balanced,high_protein"
    },
    {
        "name": "Turkey Meatballs with Spaghetti Squash",
        "recipe": "turkey_meatballs_spaghetti_squash",
        "minutes": 35,
        "description": "Lean turkey meatballs over roasted spaghetti squash with tomato sauce.",
        "ingredients": "ground turkey, spaghetti squash, tomato sauce, herbs, olive oil",
        "steps": "Roast squash, bake meatballs, top squash strands with sauce and meatballs.",
        "calories": 446,
        "proteins": 36,
        "carbs": 35,
        "fats": 18,
        "sugars": 10,
        "tags": "lunch,dinner,low_carb,high_protein"
    },
    {
        "name": "BBQ Chicken Sweet Potato Bowl",
        "recipe": "bbq_chicken_sweet_potato_bowl",
        "minutes": 30,
        "description": "BBQ chicken with roasted sweet potato and greens.",
        "ingredients": "chicken breast, BBQ sauce, sweet potato, spinach, olive oil",
        "steps": "Roast sweet potato, cook chicken with BBQ sauce, serve with greens.",
        "calories": 518,
        "proteins": 38,
        "carbs": 60,
        "fats": 14,
        "sugars": 18,
        "tags": "lunch,dinner,muscle_gain"
    },
    {
        "name": "Light Chicken Caesar Salad",
        "recipe": "light_chicken_caesar_salad",
        "minutes": 15,
        "description": "Chicken Caesar salad with lighter dressing.",
        "ingredients": "chicken breast, romaine lettuce, light Caesar dressing, croutons, parmesan",
        "steps": "Grill chicken, toss with lettuce, dressing, and toppings.",
        "calories": 380,
        "proteins": 32,
        "carbs": 18,
        "fats": 18,
        "sugars": 4,
        "tags": "lunch,dinner,fat_loss,high_protein"
    },
    {
        "name": "Spicy Beef Chili with Beans",
        "recipe": "spicy_beef_chili_beans",
        "minutes": 40,
        "description": "Spicy beef chili with kidney beans and tomatoes.",
        "ingredients": "lean ground beef, kidney beans, tomato, chili powder, onion, garlic",
        "steps": "Brown beef, add remaining ingredients, simmer until thick.",
        "calories": 482,
        "proteins": 35,
        "carbs": 45,
        "fats": 18,
        "sugars": 10,
        "tags": "lunch,dinner,high_protein,comfort_food"
    },
    {
        "name": "Beef and Barley Stew",
        "recipe": "beef_barley_stew",
        "minutes": 60,
        "description": "Slow-cooked beef stew with barley and vegetables.",
        "ingredients": "lean beef cubes, barley, carrot, celery, onion, broth",
        "steps": "Brown beef, add vegetables, barley, and broth, simmer until tender.",
        "calories": 540,
        "proteins": 34,
        "carbs": 60,
        "fats": 16,
        "sugars": 9,
        "tags": "lunch,dinner,high_protein,balanced"
    },
    {
        "name": "Honey Mustard Chicken with Veggies",
        "recipe": "honey_mustard_chicken_veggies",
        "minutes": 30,
        "description": "Baked chicken with honey mustard and roasted vegetables.",
        "ingredients": "chicken breast, honey, mustard, mixed vegetables, olive oil",
        "steps": "Coat chicken in honey mustard, bake with veggies until cooked.",
        "calories": 465,
        "proteins": 38,
        "carbs": 40,
        "fats": 17,
        "sugars": 15,
        "tags": "lunch,dinner,balanced"
    },
    {
        "name": "Cajun Shrimp with Brown Rice",
        "recipe": "cajun_shrimp_brown_rice",
        "minutes": 25,
        "description": "Spicy Cajun shrimp served with brown rice and vegetables.",
        "ingredients": "shrimp, brown rice, Cajun seasoning, bell pepper, onion, olive oil",
        "steps": "Cook rice, sauté shrimp and veggies with seasoning, serve together.",
        "calories": 456,
        "proteins": 32,
        "carbs": 55,
        "fats": 12,
        "sugars": 6,
        "tags": "lunch,dinner,pescatarian,high_protein"
    },
    {
        "name": "Tuna Pasta Salad",
        "recipe": "tuna_pasta_salad",
        "minutes": 20,
        "description": "Cold pasta salad with tuna and vegetables.",
        "ingredients": "pasta, canned tuna, peas, corn, light mayo or yogurt, herbs",
        "steps": "Cook pasta, mix with tuna, veggies, and dressing, chill before serving.",
        "calories": 520,
        "proteins": 30,
        "carbs": 65,
        "fats": 14,
        "sugars": 6,
        "tags": "lunch,dinner,pescatarian"
    },
    {
        "name": "Chicken Stir-Fry Noodles",
        "recipe": "chicken_stirfry_noodles",
        "minutes": 25,
        "description": "Stir-fried chicken with vegetables and noodles.",
        "ingredients": "chicken strips, noodles, mixed vegetables, soy sauce, garlic, oil",
        "steps": "Stir-fry chicken, add veggies and cooked noodles, toss with sauce.",
        "calories": 560,
        "proteins": 34,
        "carbs": 70,
        "fats": 14,
        "sugars": 8,
        "tags": "lunch,dinner,high_carb,muscle_gain"
    },
    {
        "name": "Turkey Chili Stuffed Peppers",
        "recipe": "turkey_chili_stuffed_peppers",
        "minutes": 35,
        "description": "Bell peppers stuffed with turkey chili and baked.",
        "ingredients": "bell peppers, ground turkey, beans, tomato sauce, spices",
        "steps": "Cook turkey chili, stuff into peppers, bake until peppers are soft.",
        "calories": 420,
        "proteins": 34,
        "carbs": 35,
        "fats": 16,
        "sugars": 10,
        "tags": "lunch,dinner,high_protein,low_carb"
    },
    {
        "name": "Baked Fish Tacos",
        "recipe": "baked_fish_tacos",
        "minutes": 25,
        "description": "Baked white fish in tortillas with slaw.",
        "ingredients": "white fish fillets, tortillas, cabbage slaw, yogurt or mayo, lime",
        "steps": "Bake seasoned fish, assemble tacos with slaw and sauce.",
        "calories": 430,
        "proteins": 30,
        "carbs": 40,
        "fats": 15,
        "sugars": 6,
        "tags": "lunch,dinner,pescatarian"
    },
    {
        "name": "Greek Chicken Gyro Plate",
        "recipe": "greek_chicken_gyro_plate",
        "minutes": 30,
        "description": "Greek-style chicken with pita, salad, and tzatziki.",
        "ingredients": "marinated chicken, pita, tomato, cucumber, lettuce, tzatziki",
        "steps": "Grill chicken, serve with pita and salad, add tzatziki on the side.",
        "calories": 534,
        "proteins": 38,
        "carbs": 55,
        "fats": 18,
        "sugars": 6,
        "tags": "lunch,dinner,high_protein,balanced"
    },

    # === LUNCH / DINNER VEGETARIAN & VEGAN (7) ===
    {
        "name": "Quinoa Stuffed Peppers",
        "recipe": "quinoa_stuffed_peppers",
        "minutes": 35,
        "description": "Bell peppers stuffed with quinoa and vegetables.",
        "ingredients": "bell peppers, quinoa, tomato, onion, spices, cheese (optional)",
        "steps": "Cook quinoa, mix with veggies, stuff peppers, bake until tender.",
        "calories": 430,
        "proteins": 16,
        "carbs": 65,
        "fats": 10,
        "sugars": 10,
        "tags": "lunch,dinner,vegetarian,high_fiber"
    },
    {
        "name": "Black Bean Burrito Bowl",
        "recipe": "black_bean_burrito_bowl",
        "minutes": 25,
        "description": "Burrito bowl with black beans, rice, and veggies.",
        "ingredients": "black beans, rice, corn, tomato, lettuce, salsa",
        "steps": "Layer rice, beans, and veggies in a bowl, top with salsa.",
        "calories": 520,
        "proteins": 18,
        "carbs": 90,
        "fats": 8,
        "sugars": 9,
        "tags": "lunch,dinner,vegan,high_fiber"
    },
    {
        "name": "Vegan Tofu Coconut Curry",
        "recipe": "vegan_tofu_coconut_curry",
        "minutes": 30,
        "description": "Tofu and vegetables in coconut curry sauce with rice.",
        "ingredients": "tofu, coconut milk, curry paste, vegetables, rice, oil",
        "steps": "Cook curry sauce, add tofu and veggies, simmer, serve with rice.",
        "calories": 580,
        "proteins": 22,
        "carbs": 65,
        "fats": 24,
        "sugars": 10,
        "tags": "lunch,dinner,vegan,comfort_food"
    },
    {
        "name": "Sweet Potato Black Bean Hash",
        "recipe": "sweet_potato_black_bean_hash",
        "minutes": 25,
        "description": "Pan-fried sweet potato with black beans and spices.",
        "ingredients": "sweet potato, black beans, onion, bell pepper, oil, spices",
        "steps": "Pan-fry diced sweet potato, add veggies and beans, season and cook.",
        "calories": 440,
        "proteins": 15,
        "carbs": 70,
        "fats": 10,
        "sugars": 14,
        "tags": "lunch,dinner,vegan,high_fiber"
    },
    {
        "name": "Spinach Ricotta Stuffed Shells",
        "recipe": "spinach_ricotta_stuffed_shells",
        "minutes": 35,
        "description": "Pasta shells stuffed with spinach and ricotta in tomato sauce.",
        "ingredients": "large pasta shells, ricotta, spinach, tomato sauce, cheese",
        "steps": "Stuff cooked shells with mixture, place in sauce, bake until bubbly.",
        "calories": 538,
        "proteins": 24,
        "carbs": 70,
        "fats": 18,
        "sugars": 10,
        "tags": "lunch,dinner,vegetarian,comfort_food"
    },
    {
        "name": "Veggie Pesto Pasta",
        "recipe": "veggie_pesto_pasta",
        "minutes": 20,
        "description": "Pasta with pesto and sautéed vegetables.",
        "ingredients": "pasta, pesto, zucchini, cherry tomatoes, olive oil",
        "steps": "Cook pasta, sauté veggies, toss with pesto and pasta.",
        "calories": 520,
        "proteins": 16,
        "carbs": 75,
        "fats": 18,
        "sugars": 8,
        "tags": "lunch,dinner,vegetarian"
    },
    {
        "name": "Falafel Wrap with Tahini",
        "recipe": "falafel_wrap_tahini",
        "minutes": 20,
        "description": "Falafel in a wrap with salad and tahini sauce.",
        "ingredients": "falafel, tortilla or pita, lettuce, tomato, cucumber, tahini sauce",
        "steps": "Warm falafel and wrap, fill with salad and sauce, roll and serve.",
        "calories": 492,
        "proteins": 18,
        "carbs": 60,
        "fats": 20,
        "sugars": 6,
        "tags": "lunch,dinner,vegetarian,vegan_option"
    },

    # === SNACKS & LIGHT OPTIONS (8) ===
    {
        "name": "Strawberry Protein Smoothie",
        "recipe": "strawberry_protein_smoothie",
        "minutes": 5,
        "description": "Smoothie with strawberries and protein powder.",
        "ingredients": "strawberries, protein powder, milk or water, ice",
        "steps": "Blend all ingredients until smooth and serve.",
        "calories": 260,
        "proteins": 24,
        "carbs": 28,
        "fats": 4,
        "sugars": 18,
        "tags": "snack,drink,high_protein"
    },
    {
        "name": "Veggie Omelette Muffins",
        "recipe": "veggie_omelette_muffins",
        "minutes": 25,
        "description": "Baked egg muffins with vegetables.",
        "ingredients": "eggs, spinach, bell pepper, onion, cheese (optional)",
        "steps": "Mix ingredients, pour into muffin tin, bake until set.",
        "calories": 220,
        "proteins": 18,
        "carbs": 4,
        "fats": 14,
        "sugars": 3,
        "tags": "snack,breakfast,prep_ahead,high_protein"
    },
    {
        "name": "High-Protein Hot Chocolate",
        "recipe": "high_protein_hot_chocolate",
        "minutes": 5,
        "description": "Hot chocolate made with protein powder.",
        "ingredients": "protein powder, cocoa powder, milk or water, sweetener",
        "steps": "Heat liquid, whisk in cocoa and protein powder, sweeten to taste.",
        "calories": 180,
        "proteins": 24,
        "carbs": 12,
        "fats": 3,
        "sugars": 8,
        "tags": "snack,drink,high_protein,night_snack"
    },
    {
        "name": "Yogurt Granola Parfait",
        "recipe": "yogurt_granola_parfait",
        "minutes": 5,
        "description": "Layered parfait with yogurt, granola, and fruit.",
        "ingredients": "Greek yogurt, granola, mixed fruit, honey",
        "steps": "Layer yogurt, granola, and fruit in a glass, drizzle honey.",
        "calories": 320,
        "proteins": 15,
        "carbs": 45,
        "fats": 9,
        "sugars": 22,
        "tags": "snack,breakfast,vegetarian"
    },
    {
        "name": "Turkey Roll-Ups",
        "recipe": "turkey_roll_ups",
        "minutes": 5,
        "description": "Turkey slices rolled with cheese and veggies.",
        "ingredients": "turkey slices, cheese slices, cucumber or pepper strips",
        "steps": "Place cheese and veggies on turkey, roll tightly and slice.",
        "calories": 190,
        "proteins": 20,
        "carbs": 3,
        "fats": 11,
        "sugars": 2,
        "tags": "snack,low_carb,high_protein"
    },
    {
        "name": "Avocado Rice Cakes",
        "recipe": "avocado_rice_cakes",
        "minutes": 5,
        "description": "Rice cakes topped with mashed avocado and seasoning.",
        "ingredients": "rice cakes, avocado, salt, pepper, lemon juice",
        "steps": "Mash avocado with seasoning, spread on rice cakes.",
        "calories": 220,
        "proteins": 4,
        "carbs": 26,
        "fats": 11,
        "sugars": 1,
        "tags": "snack,vegan,healthy_fats"
    },
    {
        "name": "Cottage Cheese and Crackers",
        "recipe": "cottage_cheese_crackers",
        "minutes": 3,
        "description": "Cottage cheese served with whole-grain crackers.",
        "ingredients": "cottage cheese, whole-grain crackers",
        "steps": "Portion cottage cheese and serve with crackers.",
        "calories": 230,
        "proteins": 16,
        "carbs": 22,
        "fats": 8,
        "sugars": 4,
        "tags": "snack,high_protein,vegetarian"
    },
    {
        "name": "Trail Mix Snack Box",
        "recipe": "trail_mix_snack_box",
        "minutes": 5,
        "description": "Homemade trail mix portion.",
        "ingredients": "nuts, seeds, dried fruit, dark chocolate chips (optional)",
        "steps": "Mix ingredients and portion into a small container.",
        "calories": 220,
        "proteins": 6,
        "carbs": 20,
        "fats": 14,
        "sugars": 12,
        "tags": "snack,vegan,energy"
    },
]

ALL_MEAL_SEED_DATA = MEAL_SEED_DATA + MORE_MEAL_SEED_DATA + EVEN_MORE_MEAL_SEED_DATA

# ============================================================================
# PROMPTS
# ============================================================================

PLANNER_PROMPT = """You are a nutrition planner specialist who creates daily nutrition plans. Think carefully and pay great attention to macro numbers.

You must create a one-day meal plan that meets the user's macro and dietary targets.

TOOLS YOU CAN USE (names must match exactly):
1) recipe_semantic_search(meal_query, k) – Search for relevant recipes and return their true macros.
2) return_final_answer_tool(answer) – Return the final answer (JSON plan).

ROUTING RULES (VERY IMPORTANT):
- If the user asks for a meal/nutrition plan, you MUST NOT call get_available_exercises or generate_workout_plan_mutation.
- Even if the request is for a 7-day plan, create only a single-day meal plan.

EXECUTION POLICY:
- You MAY call tools during reasoning to gather information and choose recipes.
- The FINAL assistant message must output ONLY a single call to return_final_answer_tool with the exact JSON plan (stringified if needed).
- You may take up to 20 turns to find the answer.

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) PLAN SKELETON
   • Generate a one-day meal plan for the user. The plan should have meals that fulfill the user's daily macro targets.
   • Create a reasonable number of meals (at least 3 distinct meals/snacks).
   • Meal names in the final plan MUST be recipe names returned by recipe_semantic_search (do not invent names).

2) USE RECIPE SEMANTIC SEARCH TOOL
   • Use recipe_semantic_search to find relevant recipes with correct nutrition info.
   • You MUST ALWAYS use the macros retrieved from the tool and NOT infer your own data.
   • For each meal you include:
     - Set "name" to the exact recipe name from the tool
     - Set "quantity" to a multiplier (e.g., 1.0 for one serving, 1.5 for 1.5 servings, 0.5 for half serving)
     - Calculate macros as: quantity × base_recipe_macros

3) MACRO ADJUSTMENT (per day)
   • Sum macros for the day across all meals.
   • If totals differ from the user's daily targets, adjust the "quantity" field for meals until daily totals are within ±5% of the user's targets.
   • Respect ALL banned keywords/ingredients from context.

4) JSON MEAL PLAN (scratch-step)
   • Build JSON matching this schema (no comments):

     {
       "meals": [
         {
           "name": "Grilled Chicken & Rice",
           "quantity": 1.5,
           "calories": 700,
           "proteins": 45,
           "carbs": 60,
           "fats": 20,
           "sequence": 1
         }
       ]
     }

   • Ensure the summed macros for the day are within ±5% of the targets.
   • IMPORTANT: Every "name" MUST match a recipe name previously returned by recipe_semantic_search. 

5) TOOL CALL (FINALIZE)
   • Call return_final_answer_tool with the EXACT JSON meal plan."""

# ============================================================================
# CoT EXAMPLE GENERATION
# ============================================================================

def search_recipes_mock(query: str, k: int = 5) -> List[Dict]:
    """Search for recipes using local seed data (MOCK)."""
    query = query.lower()
    
    # Keyword-based scoring
    scored_meals = []
    for meal in ALL_MEAL_SEED_DATA:
        score = 0
        # Simple keyword matching
        if any(term in meal["name"].lower() for term in query.split()):
            score += 3
        if any(term in meal["tags"].lower() for term in query.split()):
            score += 2
        
        # Meal type bias
        if "breakfast" in query and "breakfast" in meal["tags"]:
            score += 5
        if "lunch" in query and ("lunch" in meal["tags"] or "dinner" in meal["tags"]):
            score += 5
        if "dinner" in query and ("lunch" in meal["tags"] or "dinner" in meal["tags"]):
            score += 5
        if "snack" in query and "snack" in meal["tags"]:
            score += 5
            
        if score > 0:
            scored_meals.append((score, meal))
            
    # Sort by score descending and return top K
    scored_meals.sort(key=lambda x: x[0], reverse=True)
    
    # Format results like Pinecone would
    results = []
    for _, meal in scored_meals[:k]:
        results.append({
            "id": meal["recipe"],
            "name": meal["name"],
            "calories": meal["calories"],
            "protein": meal["proteins"],
            "carbs": meal["carbs"],
            "fat": meal["fats"]
        })
    return results

def optimize_quantities(meals, targets):
    """
    Find quantities for meals to match targets within 5% tolerance.
    Uses random search + coordinate descent.
    """
    import numpy as np
    
    # targets: [cal, prot, carb, fat]
    target_vec = np.array([
        targets.get("cal", 2000),
        targets.get("prot", 150),
        targets.get("carb", 200),
        targets.get("fat", 60)
    ])
    
    # Meal macro matrix: [4 meals x 4 macros]
    meal_matrix = np.array([
        [m["base_cal"], m["base_prot"], m["base_carb"], m["base_fat"]]
        for m in meals
    ])
    
    best_loss = float('inf')
    best_quantities = np.ones(len(meals))
    
    # Try multiple random initializations
    for _ in range(500): # Increased iterations for better convergence
        # Initialize random quantities between 0.5 and 2.5
        q = np.random.uniform(0.5, 2.5, size=len(meals))
        
        current_macros = q @ meal_matrix
        
        # check errors relative to targets
        diff = current_macros - target_vec
        relative_error = diff / target_vec
        
        # Loss = sum of squared relative errors
        loss = np.sum(relative_error ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_quantities = q.copy()
        
        # If we hit the sweet spot, return immediately
        if np.all(np.abs(relative_error) < 0.05):
            return q
            
    return best_quantities

def generate_cot_example(scenario: Scenario) -> Optional[Dict[str, Any]]:
    """
    Generate a Chain-of-Thought example showing proper tool usage and reasoning.
    Returns a conversation in Qwen chat format.
    """
    # Extract targets
    cal_target = float(scenario.daily_cal_target or 2000)
    prot_target = float(scenario.daily_prot_target or 150)
    carb_target = float(scenario.daily_carb_target or 200)
    fat_target = float(scenario.daily_fat_target or 60)
    banned = scenario.banned_keywords or []
    
    targets = {
        "cal": cal_target,
        "prot": prot_target,
        "carb": carb_target,
        "fat": fat_target
    }
    
    # Define search queries based on diet
    queries = []
    if "vegan" in scenario.question.lower() or any("vegan" in kw.lower() for kw in banned):
        queries = ["vegan protein breakfast", "vegan high protein lunch", "vegan dinner", "vegan snack"]
    elif "pescatarian" in scenario.question.lower():
        queries = ["fish breakfast", "salmon lunch", "fish dinner", "healthy snack"]
    else:
        # Mix of profiles to help optimization
        queries = ["high protein breakfast", "chicken lunch", "pasta dinner", "healthy snack"]
    
    # Try to find a valid plan (retry loop)
    max_retries = 20 # Increased retries
    for retry in range(max_retries):
        selected_recipes = []
        all_searches = []
        
        # Search for recipes using MOCK function
        current_queries = queries
        
        for query in current_queries:
            # Use mock search which searches ALL_MEAL_SEED_DATA
            recipes = search_recipes_mock(query, k=15) # Get top 15 matches
            
            if recipes:
                # Filter banned
                filtered = [
                    r for r in recipes
                    if not any(banned_kw.lower() in r["name"].lower() for banned_kw in banned)
                ]
                if filtered:
                    # Pick a random one from top 5 to ensure variety
                    choice = random.choice(filtered[:5])
                    selected_recipes.append(choice)
                    all_searches.append((query, filtered))
        
        if len(selected_recipes) < 3:
            continue
            
        # Prepare for optimization
        # Use first 4 selected recipes (or less if we found less)
        meals_to_optimize = selected_recipes[:4]
        
        meals_data = []
        for i, recipe in enumerate(meals_to_optimize):
            meals_data.append({
                "name": recipe["name"],
                "base_cal": float(recipe["calories"]),
                "base_prot": float(recipe["protein"]),
                "base_carb": float(recipe["carbs"]),
                "base_fat": float(recipe["fat"]),
                "sequence": i + 1
            })
            
        # Optimize quantities
        quantities = optimize_quantities(meals_data, targets)
        
        # Verify result
        total_cal = sum(q * m["base_cal"] for q, m in zip(quantities, meals_data))
        total_prot = sum(q * m["base_prot"] for q, m in zip(quantities, meals_data))
        total_carb = sum(q * m["base_carb"] for q, m in zip(quantities, meals_data))
        total_fat = sum(q * m["base_fat"] for q, m in zip(quantities, meals_data))
        
        # Check tolerance (5%)
        tol = 0.05
        if (abs(total_cal - cal_target)/cal_target <= tol and
            abs(total_prot - prot_target)/prot_target <= tol and
            abs(total_carb - carb_target)/carb_target <= tol and
            abs(total_fat - fat_target)/fat_target <= tol):
            
            # Found a valid plan!
            final_meals = []
            for q, meal in zip(quantities, meals_data):
                qty = round(float(q), 2)
                final_meals.append({
                    "name": meal["name"],
                    "quantity": qty,
                    "calories": round(qty * meal["base_cal"]),
                    "proteins": round(qty * meal["base_prot"], 1),
                    "carbs": round(qty * meal["base_carb"], 1),
                    "fats": round(qty * meal["base_fat"], 1),
                    "sequence": meal["sequence"]
                })
                
            return construct_conversation(scenario, targets, banned, all_searches, final_meals)
            
    print(f"Failed to find valid plan for scenario {scenario.id} after {max_retries} retries")
    return None

def construct_conversation(scenario, targets, banned, all_searches, final_meals):
    """Build the conversation messages dictionary."""
    
    cal_target = targets["cal"]
    prot_target = targets["prot"]
    carb_target = targets["carb"]
    fat_target = targets["fat"]

    # Build CoT conversation in Qwen chat format
    # Focus on teaching reasoning process and final answer format
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": PLANNER_PROMPT
    })
    
    # User message
    messages.append({
        "role": "user",
        "content": scenario.question
    })
    
    # Assistant reasoning - Chain of Thought
    # Step 1: Analyze requirements
    reasoning = "I need to create a one-day meal plan that meets the user's macro targets.\n\n"
    reasoning += f"Target macros:\n"
    reasoning += f"- Calories: {cal_target}\n"
    reasoning += f"- Protein: {prot_target}g\n"
    reasoning += f"- Carbs: {carb_target}g\n"
    reasoning += f"- Fat: {fat_target}g\n"
    if banned:
        reasoning += f"- Banned ingredients: {', '.join(banned)}\n"
    
    reasoning += "\nI'll need to search for recipes that match these requirements. Let me start by searching for different meal types.\n\n"
    
    # Step 2: Show tool usage reasoning (LangGraph will handle actual calls)
    reasoning += "I'll use recipe_semantic_search to find suitable recipes:\n"
    for i, (query, recipes) in enumerate(all_searches[:4], 1):
        reasoning += f"{i}. Searching for: \"{query}\"\n"
        if recipes:
            reasoning += f"   Found: {recipes[0]['name']} ({recipes[0]['calories']} cal, {recipes[0]['protein']}g protein)\n"
    reasoning += "\n"
    
    # Step 3: Show selection and quantity calculation reasoning
    reasoning += "Now I'll select recipes and calculate quantities to meet macro targets:\n\n"
    
    for i, meal in enumerate(final_meals, 1):
        base_cal = meal['calories'] / meal['quantity']
        base_prot = meal['proteins'] / meal['quantity']
        reasoning += f"Meal {i}: {meal['name']}\n"
        reasoning += f"  Base recipe: {base_cal:.0f} cal, {base_prot:.1f}g protein, {meal['carbs']/meal['quantity']:.1f}g carbs, {meal['fats']/meal['quantity']:.1f}g fat\n"
        reasoning += f"  Quantity multiplier: {meal['quantity']}x\n"
        reasoning += f"  Adjusted macros: {meal['calories']} cal, {meal['proteins']}g protein, {meal['carbs']}g carbs, {meal['fats']}g fat\n\n"
    
    # Step 4: Verify totals
    total_cal_final = sum(m["calories"] for m in final_meals)
    total_prot_final = sum(m["proteins"] for m in final_meals)
    total_carb_final = sum(m["carbs"] for m in final_meals)
    total_fat_final = sum(m["fats"] for m in final_meals)
    
    reasoning += "Daily totals verification:\n"
    reasoning += f"- Calories: {total_cal_final} / {cal_target} (diff: {abs(total_cal_final-cal_target)/cal_target*100:.1f}%)\n"
    reasoning += f"- Protein: {total_prot_final}g / {prot_target}g (diff: {abs(total_prot_final-prot_target)/prot_target*100:.1f}%)\n"
    reasoning += f"- Carbs: {total_carb_final}g / {carb_target}g (diff: {abs(total_carb_final-carb_target)/carb_target*100:.1f}%)\n"
    reasoning += f"- Fat: {total_fat_final}g / {fat_target}g (diff: {abs(total_fat_final-fat_target)/fat_target*100:.1f}%)\n\n"
    
    # Check if within tolerance
    cal_diff = abs(total_cal_final - cal_target) / cal_target if cal_target > 0 else 0
    prot_diff = abs(total_prot_final - prot_target) / prot_target if prot_target > 0 else 0
    carb_diff = abs(total_carb_final - carb_target) / carb_target if carb_target > 0 else 0
    fat_diff = abs(total_fat_final - fat_target) / fat_target if fat_target > 0 else 0

    if cal_diff <= 0.05 and prot_diff <= 0.05 and carb_diff <= 0.05 and fat_diff <= 0.05:
        reasoning += "✓ All macros are within ±5% tolerance. The plan is ready.\n\n"
    else:
        reasoning += "⚠ Need to adjust quantities further to meet targets.\n\n"
    
    # Step 5: Final answer format
    reasoning += "Final meal plan:\n"
    reasoning += json.dumps({"meals": final_meals}, indent=2)
    reasoning += "\n\nI'll now return this plan using return_final_answer_tool."
    
    messages.append({
        "role": "assistant",
        "content": reasoning
    })
    
    return {
        "messages": messages,
        "scenario_id": scenario.id
    }

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_training_dataset(scenarios: List[Scenario], max_examples: int = MAX_EXAMPLES) -> Dataset:
    """Generate CoT training examples from scenarios."""
    print(f"Generating {max_examples} CoT training examples...")
    
    examples = []
    random.seed(SEED)
    shuffled = random.sample(scenarios, min(len(scenarios), max_examples * 2))
    
    for scenario in tqdm(shuffled, desc="Generating examples"):
        if scenario.split != "train":
            continue
        
        example = generate_cot_example(scenario)
        if example:
            examples.append(example)
            if len(examples) >= max_examples:
                break
    
    print(f"Generated {len(examples)} valid CoT examples")
    return Dataset.from_list(examples)

# ============================================================================
# DATASET VERIFICATION
# ============================================================================

def verify_dataset(dataset: Dataset):
    """
    Verify that the generated CoT examples actually solve the task.
    Checks:
    1. Valid JSON schema
    2. Macros within ±5% tolerance
    3. Correct calculation of totals
    """
    print("\n🔍 Verifying generated dataset...")
    
    total_examples = len(dataset)
    valid_schema_count = 0
    macros_passed_count = 0
    parsing_errors = 0
    
    print(f"Checking {total_examples} examples...")
    
    for i, example in enumerate(dataset):
        messages = example["messages"]
        last_message = messages[-1]
        
        if last_message["role"] != "assistant":
            print(f"❌ Example {i}: Last message is not from assistant")
            continue
            
        content = last_message["content"]
        
        # 1. Extract Targets from reasoning text
        # Pattern: "- Calories: 2300"
        try:
            cal_target = float(re.search(r"- Calories: (\d+)", content).group(1))
            prot_target = float(re.search(r"- Protein: (\d+(?:\.\d+)?)", content).group(1))
            carb_target = float(re.search(r"- Carbs: (\d+(?:\.\d+)?)", content).group(1))
            fat_target = float(re.search(r"- Fat: (\d+(?:\.\d+)?)", content).group(1))
        except (AttributeError, ValueError):
            print(f"⚠️ Example {i}: Could not parse targets from reasoning")
            parsing_errors += 1
            continue
            
        # 2. Extract Final JSON
        try:
            # Look for the return_final_answer_tool call or just the JSON block
            json_str = re.search(r"return_final_answer_tool\(answer=(.*?)\)", content, re.DOTALL)
            if not json_str:
                # Fallback: find last JSON block
                json_str = re.search(r"(\{.*\})", content, re.DOTALL)
                if not json_str:
                    raise ValueError("No JSON found")
                payload = json.loads(json_str.group(1))
            else:
                payload = json.loads(json_str.group(1))
            
            if "meals" not in payload:
                raise ValueError("Missing 'meals' key")
                
            valid_schema_count += 1
            
            # 3. Verify Macros
            meals = payload["meals"]
            total_cal = sum(m["calories"] for m in meals)
            total_prot = sum(m["proteins"] for m in meals)
            total_carb = sum(m["carbs"] for m in meals)
            total_fat = sum(m["fats"] for m in meals)
            
            # Tolerance check (5%)
            tol = 0.051 # slightly generous for rounding
            
            cal_diff = abs(total_cal - cal_target) / cal_target if cal_target > 0 else 0
            prot_diff = abs(total_prot - prot_target) / prot_target if prot_target > 0 else 0
            carb_diff = abs(total_carb - carb_target) / carb_target if carb_target > 0 else 0
            fat_diff = abs(total_fat - fat_target) / fat_target if fat_target > 0 else 0
            
            if cal_diff <= tol and prot_diff <= tol and carb_diff <= tol and fat_diff <= tol:
                macros_passed_count += 1
            else:
                print(f"❌ Example {i} Failed Macros:")
                print(f"   Cal: {total_cal}/{cal_target} ({cal_diff:.1%})")
                print(f"   Prot: {total_prot}/{prot_target} ({prot_diff:.1%})")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Example {i}: Invalid JSON or Schema - {e}")
            continue
    
    print("\n📊 Verification Results:")
    print(f"   Total Examples: {total_examples}")
    print(f"   Valid Schema:   {valid_schema_count} ({valid_schema_count/total_examples:.1%})")
    print(f"   Macros Passed:  {macros_passed_count} ({macros_passed_count/total_examples:.1%})")
    print(f"   Parsing Errors: {parsing_errors}")
    
    if macros_passed_count / total_examples < 0.95:
        print("\n⚠️  Warning: Less than 95% of examples pass macro checks. You may want to refine the generation logic.")
    else:
        print("\n✅ Dataset quality looks good!")

# ============================================================================
# FINE-TUNING
# ============================================================================

def format_qwen_chat(messages: List[Dict]) -> str:
    """Format messages in Qwen2.5 chat format."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            formatted += f"<|im_start|>tool\n{content}<|im_end|>\n"
    
    return formatted

def fine_tune_model(dataset: Dataset):
    """Fine-tune the model using unsloth."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("❌ Unsloth not installed.")
        print("\n📦 To install unsloth, run:")
        print("   pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        print("\n   Or for CPU-only:")
        print("   pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git'")
        sys.exit(1)
    
    print(f"🤖 Loading model: {BASE_MODEL_NAME}")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"   Training examples: {len(dataset)}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Fine-tuning will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Load model with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detection
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )
    
    # Enable LoRA for efficient fine-tuning
    print("🔧 Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )
    
    # Format dataset
    print("📝 Formatting dataset...")
    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            text = format_qwen_chat(messages)
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments
    print("⚙️  Setting up training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=5,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir="./outputs",
            save_strategy="epoch",
            report_to="none",  # Disable wandb/tensorboard for simplicity
        ),
    )
    
    print("🚀 Starting fine-tuning...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE} (gradient accumulation: {GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    
    trainer.train()
    
    # Save model
    print(f"\n💾 Saving fine-tuned model to: {OUTPUT_MODEL_NAME}")
    model.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"📁 Model saved to: {OUTPUT_MODEL_NAME}")
    print(f"\n📝 Next steps:")
    print(f"   1. Update BASE_MODEL_NAME in rag_fitnessrl_v2.py to: {OUTPUT_MODEL_NAME}")
    print(f"   2. Run RL training: python scripts/rag_fitnessrl_v2.py")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load scenarios
    dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    dataset = load_dataset("json", data_files=str(dataset_path))
    training_scenarios = dataset["train"]
    
    # Process scenarios (same as rag_fitnessrl_v2.py)
    def one_day_meal_question(example):
        one_day_prompt = "generate a one day meal plan for user that match its macros and diet"
        example['input_question'] = f"{one_day_prompt}"
        return example
    
    def combine_question_and_context(example):
        context_str = json.dumps(example['context'])
        example['question'] = f"{example['input_question']} Context: {context_str}"
        return example
    
    def convert_val_to_test(example):
        if example['split'] == 'val':
            example['split'] = 'test'
        return example
    
    def get_target_nutrition_data(example):
        example['daily_cal_target'] = example['context']['daily_cal_target']
        example['daily_prot_target'] = example['context']['daily_prot_target']
        example['daily_carb_target'] = example['context']['daily_carb_target']
        example['daily_fat_target'] = example['context']['daily_fat_target']
        example['banned_keywords'] = example['context']['banned_keywords']
        return example
    
    training_scenarios = training_scenarios.map(one_day_meal_question)
    training_scenarios = training_scenarios.map(combine_question_and_context)
    training_scenarios = training_scenarios.map(convert_val_to_test)
    training_scenarios = training_scenarios.map(get_target_nutrition_data)
    
    scenarios_list = []
    for example in training_scenarios:
        scenario = Scenario(
            question=example['question'],
            split=example['split'],
            id=str(example['id']),
            daily_cal_target=example['daily_cal_target'],
            daily_prot_target=example['daily_prot_target'],
            daily_carb_target=example['daily_carb_target'],
            daily_fat_target=example['daily_fat_target'],
            banned_keywords=example['banned_keywords']
        )
        scenarios_list.append(scenario)
    
    print(f"Loaded {len(scenarios_list)} scenarios")
    
    # Generate training dataset
    train_dataset = generate_training_dataset(scenarios_list, max_examples=MAX_EXAMPLES)
    
    # Verify dataset quality
    verify_dataset(train_dataset)
    
    # Save to disk (optional)
    # train_dataset.save_to_disk("train_dataset.jsonl")
    
    # Fine-tune model
    # fine_tune_model(train_dataset)

if __name__ == "__main__":
    main()

