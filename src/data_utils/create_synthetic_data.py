data = [
  {
    "context": {
      "userId": "userT1",
      "age": 28,
      "sex": "female",
      "height_cm": 160,
      "weight_kg": 55,
      "goal": "muscle_gain",
      "activity": "moderate",
      "dietary_prefs": "no eggs, pescatarian (fish ok)",
      "equipment": "gym free weights",
      "experience": "beginner",
      "daily_cal_target": 2300,
      "daily_prot_target": 136,
      "daily_carb_target": 200,
      "daily_fat_target": 50,
      "banned_keywords": ["egg", "shellfish"]
    },
    "input_prompt": "Create a 7-day nutrition plan for this user (no eggs), matching her macro targets as closely as possible.",
    "target_output": {
      "dailyMealPlans": [
        {
          "day": 0,
          "meals": [
            {
              "name": "Salmon & Avocado Toast",
              "recipe": "rec_T101",
              "description": "Smoked salmon, avocado, whole grain toast, greens",
              "calories": 500,
              "proteins": 28,
              "carbs": 40,
              "fats": 20,
              "sequence": 1,
              "day": 0
            },
            {
              "name": "Greek Yogurt & Berries",
              "recipe": "rec_T102",
              "description": "Nonfat Greek yogurt with mixed berries & walnuts",
              "calories": 300,
              "proteins": 22,
              "carbs": 30,
              "fats": 8,
              "sequence": 2,
              "day": 0
            },
            {
              "name": "Lentil Salad with Veggies & Feta",
              "recipe": "rec_T103",
              "description": "Lentils, chickpeas, cucumber, tomato, feta, olive oil",
              "calories": 650,
              "proteins": 35,
              "carbs": 80,
              "fats": 20,
              "sequence": 3,
              "day": 0
            },
            {
              "name": "Protein Smoothie (Pea)",
              "recipe": "rec_T104",
              "description": "Pea protein, banana, spinach, almond milk",
              "calories": 300,
              "proteins": 25,
              "carbs": 35,
              "fats": 7,
              "sequence": 4,
              "day": 0
            },
            {
              "name": "Grilled Trout & Quinoa",
              "recipe": "rec_T105",
              "description": "Trout fillet, quinoa, roasted vegetables",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 10,
              "sequence": 5,
              "day": 0
            }
          ]
        },
        {
          "day": 1,
          "meals": [
            {
              "name": "Overnight Oats & Peanut Butter",
              "recipe": "rec_T111",
              "description": "Rolled oats, peanut butter, chia, soy milk",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 18,
              "sequence": 1,
              "day": 1
            },
            {
              "name": "Soy Yogurt + Nuts & Berries",
              "recipe": "rec_T112",
              "description": "Soy yogurt, almonds, mixed berries",
              "calories": 300,
              "proteins": 20,
              "carbs": 30,
              "fats": 8,
              "sequence": 2,
              "day": 1
            },
            {
              "name": "Chickpea Curry & Rice",
              "recipe": "rec_T113",
              "description": "Chickpea masala, basmati rice, vegetables",
              "calories": 700,
              "proteins": 30,
              "carbs": 100,
              "fats": 20,
              "sequence": 3,
              "day": 1
            },
            {
              "name": "Protein Bar + Apple",
              "recipe": "rec_T114",
              "description": "Vegan protein bar and apple",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 7,
              "sequence": 4,
              "day": 1
            },
            {
              "name": "Salmon & Sweet Potato",
              "recipe": "rec_T115",
              "description": "Grilled salmon, sweet potato, asparagus",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 1
            }
          ]
        },
        {
          "day": 2,
          "meals": [
            {
              "name": "Chia Pudding & Berries",
              "recipe": "rec_T121",
              "description": "Chia seeds, coconut milk, berries, flax",
              "calories": 400,
              "proteins": 15,
              "carbs": 40,
              "fats": 18,
              "sequence": 1,
              "day": 2
            },
            {
              "name": "Protein Bar & Orange",
              "recipe": "rec_T122",
              "description": "Vegan protein bar, orange",
              "calories": 300,
              "proteins": 20,
              "carbs": 35,
              "fats": 8,
              "sequence": 2,
              "day": 2
            },
            {
              "name": "Quinoa & Black Bean Bowl",
              "recipe": "rec_T123",
              "description": "Quinoa, black beans, corn, peppers, avocado",
              "calories": 650,
              "proteins": 28,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 2
            },
            {
              "name": "Smoothie Snack",
              "recipe": "rec_T124",
              "description": "Pea protein, frozen berries, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 2
            },
            {
              "name": "Grilled Cod & Rice",
              "recipe": "rec_T125",
              "description": "Cod fillet, brown rice, steamed greens",
              "calories": 550,
              "proteins": 28,
              "carbs": 95,
              "fats": 10,
              "sequence": 5,
              "day": 2
            }
          ]
        },
        {
          "day": 3,
          "meals": [
            {
              "name": "Smoothie Bowl + Granola",
              "recipe": "rec_T131",
              "description": "Pea protein, banana, berries, granola",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 15,
              "sequence": 1,
              "day": 3
            },
            {
              "name": "Hummus & Pita + Veggies",
              "recipe": "rec_T132",
              "description": "Hummus, veggies, whole wheat pita",
              "calories": 300,
              "proteins": 12,
              "carbs": 45,
              "fats": 10,
              "sequence": 2,
              "day": 3
            },
            {
              "name": "Salmon Poke Bowl",
              "recipe": "rec_T133",
              "description": "Salmon cubes, sushi rice, edamame, avocado, cucumber",
              "calories": 700,
              "proteins": 35,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 3
            },
            {
              "name": "Protein Shake (Plant)",
              "recipe": "rec_T134",
              "description": "Plant protein, spinach, pear, almond milk",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 3
            },
            {
              "name": "Tofu & Sweet Potato Bowl",
              "recipe": "rec_T135",
              "description": "Tofu, roasted sweet potato, broccoli",
              "calories": 550,
              "proteins": 24,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 3
            }
          ]
        },
        {
          "day": 4,
          "meals": [
            {
              "name": "Overnight Oats & Nut Butter",
              "recipe": "rec_T141",
              "description": "Oats, almond butter, soy milk, berries",
              "calories": 500,
              "proteins": 22,
              "carbs": 60,
              "fats": 18,
              "sequence": 1,
              "day": 4
            },
            {
              "name": "Soy Yogurt & Seeds",
              "recipe": "rec_T142",
              "description": "Soy yogurt, chia, flax, berries",
              "calories": 300,
              "proteins": 18,
              "carbs": 30,
              "fats": 12,
              "sequence": 2,
              "day": 4
            },
            {
              "name": "Chickpea Pasta & Veggies",
              "recipe": "rec_T143",
              "description": "Chickpea pasta, tomato sauce, vegetables",
              "calories": 700,
              "proteins": 32,
              "carbs": 90,
              "fats": 18,
              "sequence": 3,
              "day": 4
            },
            {
              "name": "Green Smoothie",
              "recipe": "rec_T144",
              "description": "Pea protein, spinach, apple, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 4
            },
            {
              "name": "Grilled Salmon & Veggies",
              "recipe": "rec_T145",
              "description": "Salmon fillet, asparagus, wild rice",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 4
            }
          ]
        },
        {
          "day": 5,
          "meals": [
            {
              "name": "Chia Pudding + Banana",
              "recipe": "rec_T151",
              "description": "Chia seeds, oat milk, banana, nuts",
              "calories": 400,
              "proteins": 15,
              "carbs": 45,
              "fats": 18,
              "sequence": 1,
              "day": 5
            },
            {
              "name": "Protein Bar & Mixed Fruit",
              "recipe": "rec_T152",
              "description": "Vegan protein bar and fruit",
              "calories": 300,
              "proteins": 20,
              "carbs": 35,
              "fats": 8,
              "sequence": 2,
              "day": 5
            },
            {
              "name": "Tofu Burrito Bowl",
              "recipe": "rec_T153",
              "description": "Tofu, beans, rice, vegetables, salsa",
              "calories": 700,
              "proteins": 35,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 5
            },
            {
              "name": "Smoothie Snack",
              "recipe": "rec_T154",
              "description": "Pea protein, spinach, pear, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 5
            },
            {
              "name": "Grilled Trout & Quinoa",
              "recipe": "rec_T155",
              "description": "Trout fillet, quinoa, greens",
              "calories": 550,
              "proteins": 28,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 5
            }
          ]
        },
        {
          "day": 6,
          "meals": [
            {
              "name": "Smoothie Bowl & Granola",
              "recipe": "rec_T161",
              "description": "Pea protein, fruits, granola, seeds",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 15,
              "sequence": 1,
              "day": 6
            },
            {
              "name": "Hummus & Veggies + Crackers",
              "recipe": "rec_T162",
              "description": "Hummus, carrot, cucumber, whole grain crackers",
              "calories": 300,
              "proteins": 12,
              "carbs": 40,
              "fats": 10,
              "sequence": 2,
              "day": 6
            },
            {
              "name": "Salmon Poke Bowl",
              "recipe": "rec_T163",
              "description": "Salmon, sushi rice, edamame, seaweed, cucumber",
              "calories": 700,
              "proteins": 35,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 6
            },
            {
              "name": "Protein Shake",
              "recipe": "rec_T164",
              "description": "Plant protein, banana, almond milk",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 6
            },
            {
              "name": "Quinoa & Veggie Bowl",
              "recipe": "rec_T165",
              "description": "Quinoa, mixed vegetables, olive oil",
              "calories": 550,
              "proteins": 24,
              "carbs": 90,
              "fats": 12,
              "sequence": 5,
              "day": 6
            }
          ]
        }
      ]
    }
  },
  {
    "context": {
      "userId": "userT2",
      "age": 40,
      "sex": "male",
      "height_cm": 175,
      "weight_kg": 70,
      "goal": "fat_loss",
      "activity": "light",
      "dietary_prefs": "omnivore, no eggs",
      "equipment": "gym + machines",
      "experience": "intermediate"
    },
    "input_prompt": "Generate a 7-day nutrition plan (no eggs), meeting macro targets with minimal deviation.",
    "target_output": {
      "dailyMealPlans": [
        {
          "day": 0,
          "meals": [
            {
              "name": "Tuna & Avocado Wrap",
              "recipe": "rec_T201",
              "description": "Tuna, avocado, greens, whole grain wrap",
              "calories": 450,
              "proteins": 35,
              "carbs": 45,
              "fats": 15,
              "sequence": 1,
              "day": 0
            },
            {
              "name": "Greek Yogurt & Nuts",
              "recipe": "rec_T202",
              "description": "Greek yogurt (non-fat), mixed nuts",
              "calories": 300,
              "proteins": 22,
              "carbs": 25,
              "fats": 12,
              "sequence": 2,
              "day": 0
            },
            {
              "name": "Chicken-Free “Fish” & Rice Bowl",
              "recipe": "rec_T203",
              "description": "Grilled fish substitute, jasmine rice, vegetables",
              "calories": 600,
              "proteins": 35,
              "carbs": 80,
              "fats": 15,
              "sequence": 3,
              "day": 0
            },
            {
              "name": "Protein Smoothie (Whey or Plant)",
              "recipe": "rec_T204",
              "description": "Protein powder, banana, oat milk, spinach",
              "calories": 300,
              "proteins": 25,
              "carbs": 35,
              "fats": 7,
              "sequence": 4,
              "day": 0
            },
            {
              "name": "Salmon & Sweet Potato",
              "recipe": "rec_T205",
              "description": "Grilled salmon, baked sweet potato, veggies",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 15,
              "sequence": 5,
              "day": 0
            }
          ]
        },
        {
          "day": 1,
          "meals": [
            {
              "name": "Oatmeal + Peanut Butter & Banana",
              "recipe": "rec_T211",
              "description": "Rolled oats, peanut butter, banana",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 18,
              "sequence": 1,
              "day": 1
            },
            {
              "name": "Cottage Cheese & Fruit",
              "recipe": "rec_T212",
              "description": "Low-fat cottage cheese, berries",
              "calories": 300,
              "proteins": 24,
              "carbs": 30,
              "fats": 6,
              "sequence": 2,
              "day": 1
            },
            {
              "name": "Quinoa & Chickpea Salad",
              "recipe": "rec_T213",
              "description": "Quinoa, chickpeas, mixed vegetables, olive oil",
              "calories": 650,
              "proteins": 28,
              "carbs": 90,
              "fats": 15,
              "sequence": 3,
              "day": 1
            },
            {
              "name": "Protein Bar & Apple",
              "recipe": "rec_T214",
              "description": "Vegan protein bar and apple",
              "calories": 300,
              "proteins": 20,
              "carbs": 30,
              "fats": 7,
              "sequence": 4,
              "day": 1
            },
            {
              "name": "Grilled Trout & Veggies",
              "recipe": "rec_T215",
              "description": "Trout fillet, steamed vegetables, brown rice",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 1
            }
          ]
        },
        {
          "day": 2,
          "meals": [
            {
              "name": "Chia Pudding + Berries",
              "recipe": "rec_T221",
              "description": "Chia seeds, almond milk, berries, nuts",
              "calories": 400,
              "proteins": 15,
              "carbs": 40,
              "fats": 18,
              "sequence": 1,
              "day": 2
            },
            {
              "name": "Protein Bar & Orange",
              "recipe": "rec_T222",
              "description": "Vegan protein bar, orange",
              "calories": 300,
              "proteins": 20,
              "carbs": 35,
              "fats": 8,
              "sequence": 2,
              "day": 2
            },
            {
              "name": "Black Bean & Rice Bowl",
              "recipe": "rec_T223",
              "description": "Black beans, rice, peppers, avocado",
              "calories": 650,
              "proteins": 28,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 2
            },
            {
              "name": "Smoothie Snack",
              "recipe": "rec_T224",
              "description": "Protein powder, berries, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 2
            },
            {
              "name": "Grilled Cod & Couscous",
              "recipe": "rec_T225",
              "description": "Cod, couscous, roasted vegetables",
              "calories": 550,
              "proteins": 28,
              "carbs": 95,
              "fats": 10,
              "sequence": 5,
              "day": 2
            }
          ]
        },
        {
          "day": 3,
          "meals": [
            {
              "name": "Smoothie Bowl & Granola",
              "recipe": "rec_T231",
              "description": "Protein, banana, berries, granola",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 15,
              "sequence": 1,
              "day": 3
            },
            {
              "name": "Hummus & Veggies + Crackers",
              "recipe": "rec_T232",
              "description": "Hummus, carrots, cucumber, whole grain crackers",
              "calories": 300,
              "proteins": 12,
              "carbs": 40,
              "fats": 10,
              "sequence": 2,
              "day": 3
            },
            {
              "name": "Salmon & Rice Bowl",
              "recipe": "rec_T233",
              "description": "Salmon, rice, vegetables, sauce",
              "calories": 700,
              "proteins": 35,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 3
            },
            {
              "name": "Protein Shake",
              "recipe": "rec_T234",
              "description": "Protein powder, spinach, pear, almond milk",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 3
            },
            {
              "name": "Quinoa & Veg Bowl",
              "recipe": "rec_T235",
              "description": "Quinoa, mixed vegetables, olive oil",
              "calories": 550,
              "proteins": 24,
              "carbs": 90,
              "fats": 12,
              "sequence": 5,
              "day": 3
            }
          ]
        },
        {
          "day": 4,
          "meals": [
            {
              "name": "Oatmeal + Almond Butter & Berries",
              "recipe": "rec_T241",
              "description": "Oats, almond butter, berries, soy milk",
              "calories": 500,
              "proteins": 22,
              "carbs": 60,
              "fats": 18,
              "sequence": 1,
              "day": 4
            },
            {
              "name": "Soy Yogurt & Seeds",
              "recipe": "rec_T242",
              "description": "Soy yogurt, chia, flax, berries",
              "calories": 300,
              "proteins": 18,
              "carbs": 30,
              "fats": 12,
              "sequence": 2,
              "day": 4
            },
            {
              "name": "Chickpea Pasta & Veggies",
              "recipe": "rec_T243",
              "description": "Chickpea pasta, tomato sauce, vegetables",
              "calories": 700,
              "proteins": 32,
              "carbs": 90,
              "fats": 18,
              "sequence": 3,
              "day": 4
            },
            {
              "name": "Green Smoothie",
              "recipe": "rec_T244",
              "description": "Pea protein, spinach, apple, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 4
            },
            {
              "name": "Grilled Salmon & Veggies",
              "recipe": "rec_T245",
              "description": "Salmon fillet, asparagus, wild rice",
              "calories": 550,
              "proteins": 30,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 4
            }
          ]
        },
        {
          "day": 5,
          "meals": [
            {
              "name": "Chia Pudding + Banana",
              "recipe": "rec_T251",
              "description": "Chia seeds, oat milk, banana, nuts",
              "calories": 400,
              "proteins": 15,
              "carbs": 45,
              "fats": 18,
              "sequence": 1,
              "day": 5
            },
            {
              "name": "Protein Bar & Fruit",
              "recipe": "rec_T252",
              "description": "Vegan protein bar, fruit",
              "calories": 300,
              "proteins": 20,
              "carbs": 35,
              "fats": 8,
              "sequence": 2,
              "day": 5
            },
            {
              "name": "Tempeh & Veg Stir-Fry + Rice",
              "recipe": "rec_T253",
              "description": "Tempeh, stir-fried vegetables, rice",
              "calories": 700,
              "proteins": 35,
              "carbs": 95,
              "fats": 18,
              "sequence": 3,
              "day": 5
            },
            {
              "name": "Smoothie Snack",
              "recipe": "rec_T254",
              "description": "Pea protein, spinach, pear, water",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 5
            },
            {
              "name": "Grilled Cod & Quinoa",
              "recipe": "rec_T255",
              "description": "Cod fillet, quinoa, greens",
              "calories": 550,
              "proteins": 28,
              "carbs": 85,
              "fats": 12,
              "sequence": 5,
              "day": 5
            }
          ]
        },
        {
          "day": 6,
          "meals": [
            {
              "name": "Smoothie Bowl + Granola",
              "recipe": "rec_T261",
              "description": "Protein, fruits, granola, seeds",
              "calories": 500,
              "proteins": 25,
              "carbs": 60,
              "fats": 15,
              "sequence": 1,
              "day": 6
            },
            {
              "name": "Hummus & Veggies + Crackers",
              "recipe": "rec_T262",
              "description": "Hummus, carrot, cucumber, crackers",
              "calories": 300,
              "proteins": 12,
              "carbs": 40,
              "fats": 10,
              "sequence": 2,
              "day": 6
            },
            {
              "name": "Salmon Poke Bowl",
              "recipe": "rec_T263",
              "description": "Salmon, rice, edamame, seaweed, cucumber",
              "calories": 700,
              "proteins": 35,
              "carbs": 85,
              "fats": 18,
              "sequence": 3,
              "day": 6
            },
            {
              "name": "Protein Shake",
              "recipe": "rec_T264",
              "description": "Protein powder, banana, almond milk",
              "calories": 250,
              "proteins": 20,
              "carbs": 30,
              "fats": 5,
              "sequence": 4,
              "day": 6
            },
            {
              "name": "Quinoa & Veggie Bowl",
              "recipe": "rec_T265",
              "description": "Quinoa, vegetables, olive oil",
              "calories": 550,
              "proteins": 24,
              "carbs": 90,
              "fats": 12,
              "sequence": 5,
              "day": 6
            }
          ]
        }
      ]
    }
  }
]