"""
🍕 FOOD VISION AI - ULTIMATE PRO VERSION 🍕
Complete AI-powered food detection system with advanced features
Enhanced with mobile responsiveness and optimized color scheme
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from timm import create_model
import json
import time
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import io
import base64
import plotly.graph_objs as go
import plotly.express as px
import random
import os
import re
import math
from collections import defaultdict
import sqlite3
import hashlib

# ==================== CONFIGURATION ====================
IMG_SIZE = 224
NUM_CLASSES = 101
MODEL_PATH = 'food_classifier_final.pth'
LABEL_MAPPING_PATH = 'label_mapping.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a consistent color palette with harmonious combinations
COLORS = {
    "primary": "#4F46E5",      # Indigo
    "primary_light": "#818CF8", # Light Indigo
    "primary_dark": "#3730A3",  # Dark Indigo
    "secondary": "#EC4899",    # Pink
    "secondary_light": "#F472B6", # Light Pink
    "secondary_dark": "#BE185D",   # Dark Pink
    "success": "#10B981",      # Green
    "success_light": "#34D399",   # Light Green
    "success_dark": "#059669",    # Dark Green
    "warning": "#F59E0B",      # Amber
    "warning_light": "#FBBF24",    # Light Amber
    "warning_dark": "#D97706",     # Dark Amber
    "danger": "#EF4444",       # Red
    "danger_light": "#F87171",     # Light Red
    "danger_dark": "#DC2626",      # Dark Red
    "info": "#3B82F6",         # Blue
    "info_light": "#60A5FA",       # Light Blue
    "info_dark": "#2563EB",        # Dark Blue
    "dark": "#1F2937",         # Gray-800
    "dark_light": "#374151",       # Gray-700
    "light": "#F9FAFB",        # Gray-50
    "light_dark": "#F3F4F6",      # Gray-100
    "muted": "#6B7280",        # Gray-500
    "muted_light": "#9CA3AF",      # Gray-400
    "muted_dark": "#4B5563",       # Gray-600
    "accent": "#FBBF24",       # Amber-400
    "border": "#E5E7EB",       # Gray-200
    "border_dark": "#D1D5DB",      # Gray-300
    "white": "#FFFFFF",
    "black": "#000000",
    # Harmonious gradients with better color flow
    "gradient_1": "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)",  # Indigo to Purple
    "gradient_2": "linear-gradient(135deg, #EC4899 0%, #F472B6 100%)",  # Pink gradient
    "gradient_3": "linear-gradient(135deg, #3B82F6 0%, #4F46E5 100%)",  # Blue to Indigo
    "gradient_4": "linear-gradient(135deg, #10B981 0%, #059669 100%)",  # Green gradient
    "gradient_5": "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)",  # Amber gradient
    "gradient_6": "linear-gradient(135deg, #EF4444 0%, #DC2626 100%)",  # Red gradient
    "gradient_primary": "linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%)",  # Primary flow
    "gradient_soft": "linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)",  # Soft blue
    "gradient_warm": "linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%)"  # Warm yellow
}

# Initialize database for persistent storage
def init_db():
    with sqlite3.connect('food_vision.db') as conn:
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            food_name TEXT,
            calories INTEGER,
            protein REAL,
            carbs REAL,
            fat REAL,
            fiber REAL,
            sugar REAL,
            sodium REAL,
            cholesterol REAL,
            health_score INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS water_intake (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount_ml INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS exercise (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity TEXT,
            duration_minutes INTEGER,
            calories_burned INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            goal_type TEXT,
            target_value REAL,
            current_value REAL,
            deadline DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shopping_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            item_name TEXT,
            quantity TEXT,
            checked BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()

# Initialize the database
init_db()

print("="*60)
print("🚀 INITIALIZING FOOD VISION AI - ULTIMATE PRO VERSION")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: EfficientNet-B0")
print(f"Categories: {NUM_CLASSES}")

# ==================== LOAD MODEL ====================
try:
    # Check if model files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAPPING_PATH):
        with open(LABEL_MAPPING_PATH, 'r') as f:
            idx_to_label = json.load(f)
            idx_to_label = {int(k): v for k, v in idx_to_label.items()}
        print(f"✓ Loaded {len(idx_to_label)} food categories")
        
        model = create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("✓ Model loaded successfully!")
    else:
        raise FileNotFoundError("Model files not found")
        
except Exception as e:
    print(f"⚠️ Using demo model: {e}")
    # Create a dummy model for demo purposes
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3*IMG_SIZE*IMG_SIZE, NUM_CLASSES)
        
        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            return self.fc(x)
    
    model = DummyModel().to(DEVICE)
    model.eval()
    
    # Create dummy label mapping
    food_names = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
        "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
        "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
        "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
        "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
        "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
        "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
        "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
        "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
        "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
        "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
        "mussels", "nachos", "omelette", "onion_rings", "oysters",
        "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
        "pho", "pizza", "pork_chop", "poutine", "prime_rib",
        "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
        "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
        "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
    ]
    idx_to_label = {i: food_names[i] for i in range(len(food_names))}

# ==================== PREPROCESSING ====================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== COMPREHENSIVE NUTRITION DATABASE ====================
NUTRITION_DB = {
    "apple_pie": {"cal": 237, "prot": 2, "carb": 34, "fat": 11, "fiber": 2, "emoji": "🥧", "health": 6, "sugar": 15, "sodium": 200, "cholesterol": 30},
    "baby_back_ribs": {"cal": 290, "prot": 25, "carb": 0, "fat": 21, "fiber": 0, "emoji": "🍖", "health": 5, "sugar": 0, "sodium": 450, "cholesterol": 70},
    "baklava": {"cal": 334, "prot": 5, "carb": 29, "fat": 23, "fiber": 2, "emoji": "🍰", "health": 4, "sugar": 20, "sodium": 150, "cholesterol": 35},
    "beef_carpaccio": {"cal": 127, "prot": 20, "carb": 1, "fat": 5, "fiber": 0, "emoji": "🥩", "health": 7, "sugar": 0, "sodium": 100, "cholesterol": 50},
    "beef_tartare": {"cal": 215, "prot": 17, "carb": 2, "fat": 16, "fiber": 0, "emoji": "🥩", "health": 6, "sugar": 1, "sodium": 120, "cholesterol": 65},
    "beet_salad": {"cal": 89, "prot": 3, "carb": 16, "fat": 2, "fiber": 4, "emoji": "🥗", "health": 9, "sugar": 10, "sodium": 180, "cholesterol": 0},
    "beignets": {"cal": 318, "prot": 5, "carb": 35, "fat": 18, "fiber": 1, "emoji": "🍩", "health": 4, "sugar": 12, "sodium": 200, "cholesterol": 40},
    "bibimbap": {"cal": 490, "prot": 22, "carb": 70, "fat": 12, "fiber": 8, "emoji": "🍲", "health": 8, "sugar": 8, "sodium": 600, "cholesterol": 50},
    "bread_pudding": {"cal": 291, "prot": 7, "carb": 40, "fat": 12, "fiber": 1, "emoji": "🍮", "health": 5, "sugar": 20, "sodium": 250, "cholesterol": 60},
    "breakfast_burrito": {"cal": 650, "prot": 27, "carb": 72, "fat": 27, "fiber": 7, "emoji": "🌯", "health": 6, "sugar": 5, "sodium": 980, "cholesterol": 180},
    "bruschetta": {"cal": 171, "prot": 5, "carb": 22, "fat": 7, "fiber": 2, "emoji": "🥖", "health": 7, "sugar": 4, "sodium": 250, "cholesterol": 5},
    "caesar_salad": {"cal": 184, "prot": 9, "carb": 8, "fat": 13, "fiber": 2, "emoji": "🥗", "health": 7, "sugar": 2, "sodium": 500, "cholesterol": 20},
    "cannoli": {"cal": 369, "prot": 9, "carb": 37, "fat": 20, "fiber": 1, "emoji": "🍰", "health": 4, "sugar": 20, "sodium": 150, "cholesterol": 45},
    "caprese_salad": {"cal": 163, "prot": 11, "carb": 5, "fat": 11, "fiber": 1, "emoji": "🥗", "health": 8, "sugar": 3, "sodium": 300, "cholesterol": 15},
    "carrot_cake": {"cal": 415, "prot": 4, "carb": 51, "fat": 23, "fiber": 2, "emoji": "🍰", "health": 4, "sugar": 30, "sodium": 350, "cholesterol": 55},
    "ceviche": {"cal": 120, "prot": 18, "carb": 8, "fat": 2, "fiber": 1, "emoji": "🐟", "health": 9, "sugar": 3, "sodium": 400, "cholesterol": 40},
    "cheesecake": {"cal": 321, "prot": 6, "carb": 26, "fat": 23, "fiber": 1, "emoji": "🍰", "health": 3, "sugar": 20, "sodium": 300, "cholesterol": 65},
    "cheese_plate": {"cal": 368, "prot": 23, "carb": 4, "fat": 29, "fiber": 0, "emoji": "🧀", "health": 6, "sugar": 2, "sodium": 600, "cholesterol": 80},
    "chicken_curry": {"cal": 180, "prot": 15, "carb": 12, "fat": 8, "fiber": 2, "emoji": "🍛", "health": 7, "sugar": 3, "sodium": 400, "cholesterol": 30},
    "chicken_quesadilla": {"cal": 529, "prot": 27, "carb": 40, "fat": 28, "fiber": 3, "emoji": "🌮", "health": 6, "sugar": 5, "sodium": 900, "cholesterol": 70},
    "chicken_wings": {"cal": 203, "prot": 18, "carb": 0, "fat": 14, "fiber": 0, "emoji": "🍗", "health": 5, "sugar": 0, "sodium": 500, "cholesterol": 60},
    "chocolate_cake": {"cal": 352, "prot": 5, "carb": 50, "fat": 15, "fiber": 2, "emoji": "🍰", "health": 3, "sugar": 35, "sodium": 300, "cholesterol": 40},
    "chocolate_mousse": {"cal": 189, "prot": 3, "carb": 18, "fat": 12, "fiber": 2, "emoji": "🍫", "health": 4, "sugar": 15, "sodium": 50, "cholesterol": 35},
    "churros": {"cal": 312, "prot": 5, "carb": 38, "fat": 16, "fiber": 1, "emoji": "🥨", "health": 4, "sugar": 10, "sodium": 200, "cholesterol": 30},
    "clam_chowder": {"cal": 180, "prot": 9, "carb": 16, "fat": 9, "fiber": 2, "emoji": "🍲", "health": 6, "sugar": 4, "sodium": 800, "cholesterol": 30},
    "club_sandwich": {"cal": 590, "prot": 37, "carb": 45, "fat": 27, "fiber": 3, "emoji": "🥪", "health": 6, "sugar": 6, "sodium": 1200, "cholesterol": 60},
    "crab_cakes": {"cal": 160, "prot": 12, "carb": 5, "fat": 10, "fiber": 0, "emoji": "🦀", "health": 7, "sugar": 1, "sodium": 500, "cholesterol": 45},
    "creme_brulee": {"cal": 288, "prot": 5, "carb": 24, "fat": 19, "fiber": 0, "emoji": "🍮", "health": 4, "sugar": 20, "sodium": 80, "cholesterol": 120},
    "croque_madame": {"cal": 512, "prot": 28, "carb": 30, "fat": 30, "fiber": 2, "emoji": "🥪", "health": 5, "sugar": 5, "sodium": 900, "cholesterol": 120},
    "cup_cakes": {"cal": 305, "prot": 3, "carb": 45, "fat": 13, "fiber": 1, "emoji": "🧁", "health": 3, "sugar": 30, "sodium": 200, "cholesterol": 30},
    "deviled_eggs": {"cal": 124, "prot": 7, "carb": 1, "fat": 10, "fiber": 0, "emoji": "🥚", "health": 7, "sugar": 1, "sodium": 200, "cholesterol": 185},
    "donuts": {"cal": 452, "prot": 5, "carb": 51, "fat": 25, "fiber": 1, "emoji": "🍩", "health": 2, "sugar": 25, "sodium": 300, "cholesterol": 40},
    "dumplings": {"cal": 41, "prot": 2, "carb": 7, "fat": 1, "fiber": 0, "emoji": "🥟", "health": 6, "sugar": 1, "sodium": 150, "cholesterol": 5},
    "edamame": {"cal": 122, "prot": 11, "carb": 10, "fat": 5, "fiber": 5, "emoji": "🫛", "health": 9, "sugar": 3, "sodium": 10, "cholesterol": 0},
    "eggs_benedict": {"cal": 440, "prot": 20, "carb": 28, "fat": 28, "fiber": 2, "emoji": "🍳", "health": 6, "sugar": 3, "sodium": 800, "cholesterol": 260},
    "escargots": {"cal": 150, "prot": 16, "carb": 2, "fat": 8, "fiber": 0, "emoji": "🐌", "health": 7, "sugar": 1, "sodium": 300, "cholesterol": 50},
    "falafel": {"cal": 333, "prot": 13, "carb": 32, "fat": 18, "fiber": 5, "emoji": "🧆", "health": 7, "sugar": 4, "sodium": 300, "cholesterol": 0},
    "filet_mignon": {"cal": 227, "prot": 30, "carb": 0, "fat": 11, "fiber": 0, "emoji": "🥩", "health": 7, "sugar": 0, "sodium": 70, "cholesterol": 70},
    "fish_and_chips": {"cal": 585, "prot": 32, "carb": 66, "fat": 20, "fiber": 5, "emoji": "🐟", "health": 5, "sugar": 5, "sodium": 700, "cholesterol": 60},
    "foie_gras": {"cal": 462, "prot": 11, "carb": 5, "fat": 44, "fiber": 0, "emoji": "🦆", "health": 4, "sugar": 2, "sodium": 500, "cholesterol": 150},
    "french_fries": {"cal": 312, "prot": 3, "carb": 41, "fat": 15, "fiber": 4, "emoji": "🍟", "health": 3, "sugar": 0, "sodium": 210, "cholesterol": 0},
    "french_onion_soup": {"cal": 218, "prot": 8, "carb": 15, "fat": 14, "fiber": 2, "emoji": "🍲", "health": 6, "sugar": 6, "sodium": 900, "cholesterol": 20},
    "french_toast": {"cal": 293, "prot": 10, "carb": 36, "fat": 12, "fiber": 2, "emoji": "🍞", "health": 5, "sugar": 12, "sodium": 450, "cholesterol": 90},
    "fried_calamari": {"cal": 175, "prot": 15, "carb": 8, "fat": 9, "fiber": 0, "emoji": "🦑", "health": 6, "sugar": 2, "sodium": 400, "cholesterol": 100},
    "fried_rice": {"cal": 228, "prot": 5, "carb": 34, "fat": 8, "fiber": 1, "emoji": "🍚", "health": 6, "sugar": 2, "sodium": 500, "cholesterol": 10},
    "frozen_yogurt": {"cal": 127, "prot": 3, "carb": 24, "fat": 2, "fiber": 1, "emoji": "🍦", "health": 6, "sugar": 20, "sodium": 60, "cholesterol": 5},
    "garlic_bread": {"cal": 350, "prot": 9, "carb": 43, "fat": 16, "fiber": 2, "emoji": "🥖", "health": 5, "sugar": 3, "sodium": 500, "cholesterol": 10},
    "gnocchi": {"cal": 250, "prot": 5, "carb": 48, "fat": 4, "fiber": 2, "emoji": "🍝", "health": 6, "sugar": 2, "sodium": 200, "cholesterol": 10},
    "greek_salad": {"cal": 106, "prot": 4, "carb": 8, "fat": 7, "fiber": 3, "emoji": "🥗", "health": 9, "sugar": 4, "sodium": 400, "cholesterol": 10},
    "grilled_cheese_sandwich": {"cal": 399, "prot": 17, "carb": 30, "fat": 23, "fiber": 2, "emoji": "🥪", "health": 5, "sugar": 6, "sodium": 800, "cholesterol": 45},
    "grilled_salmon": {"cal": 206, "prot": 22, "carb": 0, "fat": 12, "fiber": 0, "emoji": "🐟", "health": 9, "sugar": 0, "sodium": 80, "cholesterol": 60},
    "guacamole": {"cal": 160, "prot": 2, "carb": 9, "fat": 15, "fiber": 7, "emoji": "🥑", "health": 8, "sugar": 1, "sodium": 10, "cholesterol": 0},
    "gyoza": {"cal": 64, "prot": 3, "carb": 8, "fat": 2, "fiber": 1, "emoji": "🥟", "health": 7, "sugar": 1, "sodium": 200, "cholesterol": 5},
    "hamburger": {"cal": 295, "prot": 17, "carb": 24, "fat": 14, "fiber": 2, "emoji": "🍔", "health": 5, "sugar": 6, "sodium": 500, "cholesterol": 40},
    "hot_and_sour_soup": {"cal": 91, "prot": 6, "carb": 10, "fat": 3, "fiber": 1, "emoji": "🍲", "health": 7, "sugar": 3, "sodium": 800, "cholesterol": 10},
    "hot_dog": {"cal": 290, "prot": 11, "carb": 24, "fat": 17, "fiber": 1, "emoji": "🌭", "health": 4, "sugar": 4, "sodium": 700, "cholesterol": 30},
    "huevos_rancheros": {"cal": 345, "prot": 15, "carb": 28, "fat": 19, "fiber": 5, "emoji": "🍳", "health": 7, "sugar": 5, "sodium": 700, "cholesterol": 200},
    "hummus": {"cal": 177, "prot": 5, "carb": 15, "fat": 10, "fiber": 4, "emoji": "🫘", "health": 8, "sugar": 0, "sodium": 200, "cholesterol": 0},
    "ice_cream": {"cal": 207, "prot": 4, "carb": 24, "fat": 11, "fiber": 1, "emoji": "🍦", "health": 3, "sugar": 20, "sodium": 60, "cholesterol": 30},
    "lasagna": {"cal": 312, "prot": 17, "carb": 25, "fat": 16, "fiber": 3, "emoji": "🍝", "health": 6, "sugar": 6, "sodium": 600, "cholesterol": 40},
    "lobster_bisque": {"cal": 194, "prot": 9, "carb": 11, "fat": 12, "fiber": 1, "emoji": "🦞", "health": 6, "sugar": 5, "sodium": 700, "cholesterol": 50},
    "lobster_roll_sandwich": {"cal": 436, "prot": 23, "carb": 36, "fat": 21, "fiber": 2, "emoji": "🦞", "health": 6, "sugar": 6, "sodium": 900, "cholesterol": 80},
    "macaroni_and_cheese": {"cal": 330, "prot": 13, "carb": 40, "fat": 13, "fiber": 2, "emoji": "🧀", "health": 5, "sugar": 6, "sodium": 700, "cholesterol": 35},
    "macarons": {"cal": 97, "prot": 2, "carb": 14, "fat": 4, "fiber": 1, "emoji": "🍬", "health": 4, "sugar": 10, "sodium": 30, "cholesterol": 10},
    "miso_soup": {"cal": 40, "prot": 3, "carb": 5, "fat": 1, "fiber": 1, "emoji": "🍲", "health": 8, "sugar": 2, "sodium": 800, "cholesterol": 0},
    "mussels": {"cal": 86, "prot": 12, "carb": 4, "fat": 2, "fiber": 0, "emoji": "🦪", "health": 9, "sugar": 2, "sodium": 400, "cholesterol": 30},
    "nachos": {"cal": 346, "prot": 9, "carb": 36, "fat": 19, "fiber": 4, "emoji": "🧀", "health": 4, "sugar": 3, "sodium": 600, "cholesterol": 25},
    "omelette": {"cal": 154, "prot": 11, "carb": 1, "fat": 12, "fiber": 0, "emoji": "🍳", "health": 7, "sugar": 1, "sodium": 150, "cholesterol": 200},
    "onion_rings": {"cal": 407, "prot": 5, "carb": 38, "fat": 26, "fiber": 2, "emoji": "🧅", "health": 3, "sugar": 5, "sodium": 500, "cholesterol": 0},
    "oysters": {"cal": 68, "prot": 8, "carb": 4, "fat": 2, "fiber": 0, "emoji": "🦪", "health": 9, "sugar": 1, "sodium": 200, "cholesterol": 25},
    "pad_thai": {"cal": 429, "prot": 11, "carb": 52, "fat": 19, "fiber": 3, "emoji": "🍜", "health": 6, "sugar": 10, "sodium": 1000, "cholesterol": 50},
    "paella": {"cal": 235, "prot": 14, "carb": 30, "fat": 6, "fiber": 2, "emoji": "🥘", "health": 7, "sugar": 3, "sodium": 500, "cholesterol": 30},
    "pancakes": {"cal": 227, "prot": 6, "carb": 28, "fat": 10, "fiber": 1, "emoji": "🥞", "health": 5, "sugar": 10, "sodium": 450, "cholesterol": 40},
    "panna_cotta": {"cal": 224, "prot": 5, "carb": 21, "fat": 14, "fiber": 0, "emoji": "🍮", "health": 4, "sugar": 18, "sodium": 50, "cholesterol": 60},
    "peking_duck": {"cal": 337, "prot": 19, "carb": 0, "fat": 28, "fiber": 0, "emoji": "🦆", "health": 6, "sugar": 0, "sodium": 500, "cholesterol": 90},
    "pho": {"cal": 350, "prot": 15, "carb": 45, "fat": 12, "fiber": 2, "emoji": "🍜", "health": 7, "sugar": 5, "sodium": 900, "cholesterol": 30},
    "pizza": {"cal": 266, "prot": 11, "carb": 33, "fat": 10, "fiber": 3, "emoji": "🍕", "health": 5, "sugar": 4, "sodium": 600, "cholesterol": 20},
    "pork_chop": {"cal": 231, "prot": 25, "carb": 0, "fat": 14, "fiber": 0, "emoji": "🥩", "health": 6, "sugar": 0, "sodium": 70, "cholesterol": 70},
    "poutine": {"cal": 740, "prot": 15, "carb": 83, "fat": 38, "fiber": 5, "emoji": "🍟", "health": 3, "sugar": 2, "sodium": 1200, "cholesterol": 60},
    "prime_rib": {"cal": 338, "prot": 24, "carb": 0, "fat": 27, "fiber": 0, "emoji": "🥩", "health": 5, "sugar": 0, "sodium": 100, "cholesterol": 80},
    "pulled_pork_sandwich": {"cal": 415, "prot": 29, "carb": 37, "fat": 16, "fiber": 2, "emoji": "🥪", "health": 6, "sugar": 8, "sodium": 800, "cholesterol": 60},
    "ramen": {"cal": 436, "prot": 13, "carb": 62, "fat": 14, "fiber": 3, "emoji": "🍜", "health": 6, "sugar": 5, "sodium": 1500, "cholesterol": 30},
    "ravioli": {"cal": 220, "prot": 9, "carb": 30, "fat": 7, "fiber": 2, "emoji": "🍝", "health": 6, "sugar": 4, "sodium": 400, "cholesterol": 25},
    "red_velvet_cake": {"cal": 478, "prot": 5, "carb": 71, "fat": 20, "fiber": 1, "emoji": "🍰", "health": 2, "sugar": 50, "sodium": 350, "cholesterol": 60},
    "risotto": {"cal": 174, "prot": 4, "carb": 26, "fat": 6, "fiber": 1, "emoji": "🍚", "health": 6, "sugar": 2, "sodium": 500, "cholesterol": 10},
    "samosa": {"cal": 252, "prot": 4, "carb": 24, "fat": 16, "fiber": 3, "emoji": "🥟", "health": 5, "sugar": 2, "sodium": 300, "cholesterol": 10},
    "sashimi": {"cal": 145, "prot": 24, "carb": 0, "fat": 5, "fiber": 0, "emoji": "🍣", "health": 9, "sugar": 0, "sodium": 50, "cholesterol": 40},
    "scallops": {"cal": 94, "prot": 17, "carb": 3, "fat": 1, "fiber": 0, "emoji": "🦪", "health": 9, "sugar": 1, "sodium": 200, "cholesterol": 25},
    "seaweed_salad": {"cal": 45, "prot": 2, "carb": 8, "fat": 1, "fiber": 3, "emoji": "🥗", "health": 9, "sugar": 3, "sodium": 300, "cholesterol": 0},
    "shrimp_and_grits": {"cal": 365, "prot": 20, "carb": 38, "fat": 14, "fiber": 2, "emoji": "🍤", "health": 6, "sugar": 3, "sodium": 900, "cholesterol": 100},
    "spaghetti_bolognese": {"cal": 259, "prot": 15, "carb": 31, "fat": 8, "fiber": 3, "emoji": "🍝", "health": 7, "sugar": 6, "sodium": 500, "cholesterol": 30},
    "spaghetti_carbonara": {"cal": 394, "prot": 17, "carb": 43, "fat": 17, "fiber": 2, "emoji": "🍝", "health": 5, "sugar": 4, "sodium": 700, "cholesterol": 120},
    "spring_rolls": {"cal": 150, "prot": 5, "carb": 20, "fat": 6, "fiber": 2, "emoji": "🥟", "health": 7, "sugar": 3, "sodium": 300, "cholesterol": 10},
    "steak": {"cal": 271, "prot": 25, "carb": 0, "fat": 18, "fiber": 0, "emoji": "🥩", "health": 6, "sugar": 0, "sodium": 70, "cholesterol": 70},
    "strawberry_shortcake": {"cal": 320, "prot": 4, "carb": 45, "fat": 14, "fiber": 2, "emoji": "🍰", "health": 4, "sugar": 25, "sodium": 250, "cholesterol": 35},
    "sushi": {"cal": 143, "prot": 6, "carb": 21, "fat": 4, "fiber": 1, "emoji": "🍣", "health": 8, "sugar": 3, "sodium": 300, "cholesterol": 10},
    "tacos": {"cal": 226, "prot": 9, "carb": 20, "fat": 12, "fiber": 3, "emoji": "🌮", "health": 6, "sugar": 3, "sodium": 400, "cholesterol": 25},
    "takoyaki": {"cal": 40, "prot": 2, "carb": 5, "fat": 1, "fiber": 0, "emoji": "🐙", "health": 6, "sugar": 1, "sodium": 150, "cholesterol": 5},
    "tiramisu": {"cal": 240, "prot": 5, "carb": 28, "fat": 12, "fiber": 1, "emoji": "🍰", "health": 4, "sugar": 20, "sodium": 100, "cholesterol": 60},
    "tuna_tartare": {"cal": 98, "prot": 23, "carb": 0, "fat": 1, "fiber": 0, "emoji": "🐟", "health": 9, "sugar": 0, "sodium": 50, "cholesterol": 30},
    "waffles": {"cal": 291, "prot": 8, "carb": 33, "fat": 13, "fiber": 2, "emoji": "🧇", "health": 5, "sugar": 10, "sodium": 450, "cholesterol": 50},
}

# ==================== RECIPE DATABASE ====================
RECIPE_DB = {
    "apple_pie": {
        "name": "Classic Apple Pie",
        "time": "90 min",
        "difficulty": "Medium",
        "ingredients": ["Apples", "Pie crust", "Sugar", "Cinnamon", "Butter", "Flour"],
        "steps": ["Prepare the pie crust", "Peel and slice apples", "Mix apples with sugar and cinnamon", "Fill the pie crust", "Bake at 375°F for 45-50 minutes"],
        "calories": 237,
        "servings": 8
    },
    "pizza": {
        "name": "Homemade Pizza",
        "time": "2 hours",
        "difficulty": "Medium",
        "ingredients": ["Flour", "Yeast", "Tomato sauce", "Mozzarella", "Toppings", "Olive oil"],
        "steps": ["Make dough and let rise", "Roll out and add toppings", "Bake at 475°F for 12-15 minutes", "Slice and serve"],
        "calories": 266,
        "servings": 4
    },
    "sushi": {
        "name": "California Rolls",
        "time": "45 min",
        "difficulty": "Medium",
        "ingredients": ["Sushi rice", "Nori", "Crab", "Avocado", "Cucumber", "Sesame seeds"],
        "steps": ["Season sushi rice", "Place nori on mat", "Spread rice on nori", "Add fillings", "Roll tightly and slice"],
        "calories": 143,
        "servings": 6
    },
    "salad": {
        "name": "Garden Salad",
        "time": "15 min",
        "difficulty": "Easy",
        "ingredients": ["Mixed greens", "Tomatoes", "Cucumber", "Bell peppers", "Olive oil", "Lemon"],
        "steps": ["Wash and chop vegetables", "Mix greens in bowl", "Add vegetables", "Drizzle with dressing", "Toss and serve"],
        "calories": 89,
        "servings": 4
    },
    "burger": {
        "name": "Classic Hamburger",
        "time": "20 min",
        "difficulty": "Easy",
        "ingredients": ["Ground beef", "Burger buns", "Lettuce", "Tomato", "Onion", "Cheese", "Pickles"],
        "steps": ["Form beef patties", "Season and grill", "Toast buns", "Assemble with toppings", "Serve immediately"],
        "calories": 295,
        "servings": 4
    },
    "pasta": {
        "name": "Spaghetti Carbonara",
        "time": "20 min",
        "difficulty": "Medium",
        "ingredients": ["Spaghetti", "Pancetta", "Eggs", "Parmesan", "Black pepper", "Garlic"],
        "steps": ["Cook spaghetti", "Crisp pancetta", "Mix eggs with cheese", "Toss hot pasta with egg mixture", "Add pancetta and pepper"],
        "calories": 394,
        "servings": 4
    },
    "tacos": {
        "name": "Beef Tacos",
        "time": "25 min",
        "difficulty": "Easy",
        "ingredients": ["Ground beef", "Taco shells", "Lettuce", "Tomato", "Cheese", "Sour cream", "Salsa"],
        "steps": ["Brown ground beef with spices", "Warm taco shells", "Fill shells with beef", "Add toppings", "Serve immediately"],
        "calories": 226,
        "servings": 4
    },
    "ice_cream": {
        "name": "Vanilla Ice Cream",
        "time": "4 hours",
        "difficulty": "Easy",
        "ingredients": ["Heavy cream", "Milk", "Sugar", "Egg yolks", "Vanilla"],
        "steps": ["Heat milk and cream", "Whisk yolks with sugar", "Combine and cook until thick", "Chill completely", "Churn in ice cream maker"],
        "calories": 207,
        "servings": 8
    },
    "chicken_curry": {
        "name": "Chicken Curry",
        "time": "45 min",
        "difficulty": "Medium",
        "ingredients": ["Chicken", "Onions", "Garlic", "Ginger", "Curry powder", "Coconut milk", "Rice"],
        "steps": ["Sauté onions, garlic, ginger", "Add curry powder and cook", "Add chicken and brown", "Pour in coconut milk", "Simmer until chicken is cooked"],
        "calories": 180,
        "servings": 4
    },
    "pancakes": {
        "name": "Fluffy Pancakes",
        "time": "20 min",
        "difficulty": "Easy",
        "ingredients": ["Flour", "Sugar", "Baking powder", "Milk", "Eggs", "Butter", "Maple syrup"],
        "steps": ["Mix dry ingredients", "Combine with wet ingredients", "Cook on griddle", "Flip when bubbles form", "Serve with syrup and butter"],
        "calories": 227,
        "servings": 4
    }
}

# ==================== FUN FACTS DATABASE ====================
FUN_FACTS = {
    "pizza": "🍕 Americans eat approximately 350 slices of pizza per second!",
    "sushi": "🍣 'Sushi' actually refers to the rice, not the fish!",
    "hamburger": "🍔 The world's most expensive burger costs $5,000 and includes truffles and gold leaf!",
    "ramen": "🍜 There are over 40,000 ramen shops in Japan!",
    "chicken_curry": "🍛 Curry was brought to Japan by the British Navy in the 1800s!",
    "tacos": "🌮 Tacos date back to the 18th century Mexican silver mines!",
    "ice_cream": "🍦 It takes about 50 licks to finish a single scoop of ice cream!",
    "chocolate_cake": "🍰 Chocolate was consumed as a beverage for 90% of its 4,000-year history!",
    "donuts": "🍩 Americans consume 10 billion donuts annually!",
    "steak": "🥩 The most expensive steak in the world costs $3,200 for a single cut!",
}

# ==================== MEAL HISTORY DATABASE ====================
MEAL_HISTORY = []

# ==================== USER PROFILE ====================
USER_PROFILE = {
    "name": "Food Lover",
    "email": "",
    "password_hash": "",
    "daily_calorie_goal": 2000,
    "dietary_preferences": [],
    "allergies": [],
    "favorite_foods": [],
    "height_cm": 170,
    "weight_kg": 70,
    "activity_level": "moderate",
    "water_goal_ml": 2000,
    "macro_targets": {
        "protein_percent": 30,
        "carbs_percent": 40,
        "fat_percent": 30
    }
}

# ==================== HELPER FUNCTIONS ====================
def get_nutrition_info(food_name):
    """Get comprehensive nutrition information"""
    key = food_name.lower().replace(' ', '_')
    
    if key in NUTRITION_DB:
        data = NUTRITION_DB[key]
        return {
            "emoji": data["emoji"],
            "calories": data["cal"],
            "protein": data["prot"],
            "carbs": data["carb"],
            "fat": data["fat"],
            "fiber": data["fiber"],
            "health_score": data["health"],
            "sugar": data["sugar"],
            "sodium": data["sodium"],
            "cholesterol": data["cholesterol"]
        }
    
    # Default if not found
    return {
        "emoji": "🍽️",
        "calories": 250,
        "protein": 10,
        "carbs": 30,
        "fat": 10,
        "fiber": 3,
        "health_score": 5,
        "sugar": 10,
        "sodium": 300,
        "cholesterol": 30
    }

def get_recipe_info(food_name):
    """Get recipe information"""
    key = food_name.lower().replace(' ', '_')
    
    if key in RECIPE_DB:
        return RECIPE_DB[key]
    
    # Default if not found
    return {
        "name": f"{food_name} Recipe",
        "time": "30 min",
        "difficulty": "Medium",
        "ingredients": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
        "steps": ["Step 1", "Step 2", "Step 3"],
        "calories": 250,
        "servings": 4
    }

def get_health_rating(health_score):
    """Convert health score to rating"""
    if health_score >= 8:
        return "🟢 **Excellent Choice!** Very nutritious and healthy.", COLORS["success"]
    elif health_score >= 6:
        return "🟡 **Good Choice!** Balanced and nutritious.", COLORS["warning"]
    elif health_score >= 4:
        return "🟠 **Okay in Moderation.** Enjoy occasionally.", COLORS["accent"]
    else:
        return "🔴 **Occasional Treat.** High in calories or fat.", COLORS["danger"]

def get_fun_fact(food_name):
    """Get fun fact about the food"""
    key = food_name.lower().replace(' ', '_')
    return FUN_FACTS.get(key, f"🎯 {food_name} is delicious and enjoyed worldwide!")

def add_to_meal_history(food_name, nutrition, timestamp):
    """Add food to meal history"""
    meal = {
        "food": food_name,
        "nutrition": nutrition,
        "timestamp": timestamp,
        "id": len(MEAL_HISTORY) + 1
    }
    MEAL_HISTORY.append(meal)
    
    # Also save to database
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO meals (user_id, food_name, calories, protein, carbs, fat, fiber, sugar, sodium, cholesterol, health_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                1,  # Default user ID
                food_name,
                nutrition["calories"],
                nutrition["protein"],
                nutrition["carbs"],
                nutrition["fat"],
                nutrition["fiber"],
                nutrition["sugar"],
                nutrition["sodium"],
                nutrition["cholesterol"],
                nutrition["health_score"],
                timestamp
            ))
            conn.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
    
    return meal

def get_daily_calorie_intake():
    """Calculate total calories consumed today"""
    today = date.today().strftime("%Y-%m-%d")
    total = 0
    
    # Get from database
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT SUM(calories) FROM meals 
            WHERE user_id = ? AND DATE(timestamp) = ?
            ''', (1, today))
            result = cursor.fetchone()
            
            if result and result[0]:
                total = result[0]
    except Exception as e:
        print(f"Error getting daily intake: {e}")
    
    return total

def get_weekly_nutrition_chart():
    """Generate a chart of weekly nutrition intake"""
    # Get the last 7 days
    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    
    # Initialize data for each day
    daily_data = {str(start_date + timedelta(days=i)): {"calories": 0, "protein": 0, "carbs": 0, "fat": 0} 
                  for i in range(7)}
    
    # Get data from database
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            
            for i in range(7):
                current_date = str(start_date + timedelta(days=i))
                cursor.execute('''
                SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat) FROM meals 
                WHERE user_id = ? AND DATE(timestamp) = ?
                ''', (1, current_date))
                result = cursor.fetchone()
                
                if result and result[0]:
                    daily_data[current_date]["calories"] = result[0] or 0
                    daily_data[current_date]["protein"] = result[1] or 0
                    daily_data[current_date]["carbs"] = result[2] or 0
                    daily_data[current_date]["fat"] = result[3] or 0
    except Exception as e:
        print(f"Error getting weekly data: {e}")
    
    # Create chart
    dates = list(daily_data.keys())
    calories = [daily_data[d]["calories"] for d in dates]
    protein = [daily_data[d]["protein"] for d in dates]
    carbs = [daily_data[d]["carbs"] for d in dates]
    fat = [daily_data[d]["fat"] for d in dates]
    
    # Format dates for display
    formatted_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d") for d in dates]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=formatted_dates,
        y=calories,
        mode='lines+markers',
        name='Calories',
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=formatted_dates,
        y=protein,
        mode='lines+markers',
        name='Protein (g)',
        line=dict(color=COLORS["success"], width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=formatted_dates,
        y=carbs,
        mode='lines+markers',
        name='Carbs (g)',
        line=dict(color=COLORS["warning"], width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=formatted_dates,
        y=fat,
        mode='lines+markers',
        name='Fat (g)',
        line=dict(color=COLORS["danger"], width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Weekly Nutrition Intake",
        xaxis_title="Date",
        yaxis_title="Amount",
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color=COLORS["dark"])
    )
    
    return fig

def get_meal_recommendations():
    """Get meal recommendations based on current intake"""
    daily_intake = get_daily_calorie_intake()
    remaining = USER_PROFILE["daily_calorie_goal"] - daily_intake
    
    if remaining <= 0:
        return "You've reached your daily calorie goal! Consider lighter options for the rest of the day."
    
    # Find foods that fit within remaining calories
    recommendations = []
    for key, data in NUTRITION_DB.items():
        if data["cal"] <= remaining:
            food_name = key.replace('_', ' ').title()
            recommendations.append((food_name, data["cal"], data["health"]))
    
    # Sort by health score (descending)
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    # Return top 5
    top_recommendations = recommendations[:5]
    result = f"You have {remaining} calories remaining today. Here are some healthy options:\n\n"
    
    for food, cal, health in top_recommendations:
        emoji = NUTRITION_DB[food.lower().replace(' ', '_')]["emoji"]
        result += f"- {emoji} {food} ({cal} cal)\n"
    
    return result

def export_nutrition_report():
    """Export nutrition report as CSV"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT DATE(timestamp) as date, TIME(timestamp) as time, food_name, calories, protein, carbs, fat, fiber, health_score
            FROM meals 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            ''', (1,))
            
            data = cursor.fetchall()
        
        if not data:
            return None, "No meal history to export"
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["Date", "Time", "Food", "Calories", "Protein", "Carbs", "Fat", "Fiber", "Health Score"])
        
        # Create CSV string
        csv_content = df.to_csv(index=False)
        
        # Create file on disk so Gradio can serve it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nutrition_report_{timestamp}.csv"
        with open(filename, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)
        
        # Return the filename so that Gradio can present it for download
        return filename, csv_content
    except Exception as e:
        print(f"Error exporting report: {e}")
        return None, f"Error exporting report: {str(e)}"

def compare_foods(food1, food2):
    """Compare nutrition information between two foods"""
    nutrition1 = get_nutrition_info(food1)
    nutrition2 = get_nutrition_info(food2)
    
    # Create comparison chart
    categories = ['Calories', 'Protein', 'Carbs', 'Fat', 'Fiber']
    values1 = [nutrition1["calories"], nutrition1["protein"], nutrition1["carbs"], nutrition1["fat"], nutrition1["fiber"]]
    values2 = [nutrition2["calories"], nutrition2["protein"], nutrition2["carbs"], nutrition2["fat"], nutrition2["fiber"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=food1,
        x=categories,
        y=values1,
        marker_color=COLORS["primary"]
    ))
    
    fig.add_trace(go.Bar(
        name=food2,
        x=categories,
        y=values2,
        marker_color=COLORS["secondary"]
    ))
    
    fig.update_layout(
        title=f"Nutrition Comparison: {food1} vs {food2}",
        xaxis_title="Nutrient",
        yaxis_title="Amount",
        barmode='group',
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color=COLORS["dark"])
    )
    
    # Create comparison text
    comparison_text = f"""
    ## {nutrition1["emoji"]} {food1} vs {nutrition2["emoji"]} {food2}
    
    ### Health Score
    - {food1}: {nutrition1["health_score"]}/10
    - {food2}: {nutrition2["health_score"]}/10
    
    ### Nutrition Highlights
    - **Calories**: {food1} has {abs(nutrition1["calories"] - nutrition2["calories"])} {'more' if nutrition1["calories"] > nutrition2["calories"] else 'fewer'} calories than {food2}
    - **Protein**: {food1} has {abs(nutrition1["protein"] - nutrition2["protein"])}g {'more' if nutrition1["protein"] > nutrition2["protein"] else 'less'} protein than {food2}
    - **Carbs**: {food1} has {abs(nutrition1["carbs"] - nutrition2["carbs"])}g {'more' if nutrition1["carbs"] > nutrition2["carbs"] else 'less'} carbs than {food2}
    - **Fat**: {food1} has {abs(nutrition1["fat"] - nutrition2["fat"])}g {'more' if nutrition1["fat"] > nutrition2["fat"] else 'less'} fat than {food2}
    - **Fiber**: {food1} has {abs(nutrition1["fiber"] - nutrition2["fiber"])}g {'more' if nutrition1["fiber"] > nutrition2["fiber"] else 'less'} fiber than {food2}
    
    ### Recommendation
    {'Choose ' + food1 + ' for a healthier option' if nutrition1["health_score"] > nutrition2["health_score"] else 'Choose ' + food2 + ' for a healthier option'}
    """
    
    return fig, comparison_text

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
        color = COLORS["info"]
    elif bmi < 25:
        category = "Normal weight"
        color = COLORS["success"]
    elif bmi < 30:
        category = "Overweight"
        color = COLORS["warning"]
    else:
        category = "Obese"
        color = COLORS["danger"]
    
    return round(bmi, 1), category, color

def calculate_daily_calories(height_cm, weight_kg, age, gender, activity_level):
    """Calculate daily calorie needs using Mifflin-St Jeor Equation"""
    # Convert height to meters
    height_m = height_cm / 100
    
    # Base metabolic rate
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    # Activity factor
    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    
    activity_factor = activity_factors.get(activity_level, 1.55)
    daily_calories = bmr * activity_factor
    
    return round(daily_calories)

def get_water_intake_today():
    """Get today's water intake"""
    today = date.today().strftime("%Y-%m-%d")
    
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT SUM(amount_ml) FROM water_intake 
            WHERE user_id = ? AND DATE(timestamp) = ?
            ''', (1, today))
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
    except Exception as e:
        print(f"Error getting water intake: {e}")
    
    return 0

def add_water_intake(amount_ml):
    """Add water intake to database"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO water_intake (user_id, amount_ml, timestamp)
            VALUES (?, ?, ?)
            ''', (1, amount_ml, datetime.now()))
            conn.commit()
        
        return get_water_intake_today()
    except Exception as e:
        print(f"Error adding water intake: {e}")
        return get_water_intake_today()

def get_exercise_today():
    """Get today's exercise"""
    today = date.today().strftime("%Y-%m-%d")
    
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT activity, duration_minutes, calories_burned FROM exercise 
            WHERE user_id = ? AND DATE(timestamp) = ?
            ORDER BY timestamp DESC
            ''', (1, today))
            result = cursor.fetchall()
            
            return result
    except Exception as e:
        print(f"Error getting exercise: {e}")
        return []

def add_exercise(activity, duration_minutes, calories_burned):
    """Add exercise to database"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO exercise (user_id, activity, duration_minutes, calories_burned, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (1, activity, duration_minutes, calories_burned, datetime.now()))
            conn.commit()
        
        return get_exercise_today()
    except Exception as e:
        print(f"Error adding exercise: {e}")
        return get_exercise_today()

def get_shopping_list():
    """Get shopping list from database"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, item_name, quantity, checked FROM shopping_list 
            WHERE user_id = ?
            ORDER BY checked, id
            ''', (1,))
            result = cursor.fetchall()
            
            return result
    except Exception as e:
        print(f"Error getting shopping list: {e}")
        return []

def add_to_shopping_list(item_name, quantity):
    """Add item to shopping list"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO shopping_list (user_id, item_name, quantity, checked)
            VALUES (?, ?, ?, 0)
            ''', (1, item_name, quantity))
            conn.commit()
        
        return get_shopping_list()
    except Exception as e:
        print(f"Error adding to shopping list: {e}")
        return get_shopping_list()

def toggle_shopping_item(item_id):
    """Toggle shopping item checked status"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            
            # Get current status
            cursor.execute('''
            SELECT checked FROM shopping_list 
            WHERE id = ? AND user_id = ?
            ''', (item_id, 1))
            result = cursor.fetchone()
            
            if result:
                new_status = 0 if result[0] == 1 else 1
                cursor.execute('''
                UPDATE shopping_list 
                SET checked = ? 
                WHERE id = ? AND user_id = ?
                ''', (new_status, item_id, 1))
                conn.commit()
        
        return get_shopping_list()
    except Exception as e:
        print(f"Error toggling shopping item: {e}")
        return get_shopping_list()

def delete_shopping_item(item_id):
    """Delete item from shopping list"""
    try:
        with sqlite3.connect('food_vision.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            DELETE FROM shopping_list 
            WHERE id = ? AND user_id = ?
            ''', (item_id, 1))
            conn.commit()
        
        return get_shopping_list()
    except Exception as e:
        print(f"Error deleting shopping item: {e}")
        return get_shopping_list()

def generate_meal_plan(meal_type, calorie_target):
    """Generate a meal plan based on calorie target"""
    # Filter foods by appropriate calorie range
    suitable_foods = []
    
    for key, data in NUTRITION_DB.items():
        food_name = key.replace('_', ' ').title()
        
        # Adjust calorie target based on meal type
        if meal_type == "Breakfast":
            target_range = (calorie_target * 0.2, calorie_target * 0.3)
        elif meal_type == "Lunch":
            target_range = (calorie_target * 0.3, calorie_target * 0.4)
        elif meal_type == "Dinner":
            target_range = (calorie_target * 0.3, calorie_target * 0.4)
        else:  # Snack
            target_range = (calorie_target * 0.05, calorie_target * 0.15)
        
        if target_range[0] <= data["cal"] <= target_range[1]:
            suitable_foods.append((food_name, data))
    
    # Sort by health score
    suitable_foods.sort(key=lambda x: x[1]["health"], reverse=True)
    
    # Select 3-5 options
    selected = suitable_foods[:min(5, len(suitable_foods))]
    
    if not selected:
        return f"No suitable {meal_type.lower()} options found for {calorie_target} calorie target."
    
    result = f"### {meal_type} Options (Target: {target_range[0]}-{target_range[1]} calories)\n\n"
    
    for food_name, data in selected:
        result += f"- {data['emoji']} **{food_name}** ({data['cal']} cal, Health Score: {data['health']}/10)\n"
    
    return result

def get_food_substitutions(food_name):
    """Get healthier substitutions for a food"""
    nutrition = get_nutrition_info(food_name)
    
    # Find similar foods with better health scores
    substitutions = []
    
    for key, data in NUTRITION_DB.items():
        sub_name = key.replace('_', ' ').title()
        
        # Skip if same food
        if sub_name.lower() == food_name.lower():
            continue
        
        # Check if healthier
        if data["health"] > nutrition["health_score"]:
            # Calculate similarity based on calories
            calorie_diff = abs(data["cal"] - nutrition["calories"])
            if calorie_diff <= 100:  # Within 100 calories
                substitutions.append((sub_name, data, calorie_diff))
    
    # Sort by health score and calorie similarity
    substitutions.sort(key=lambda x: (x[1]["health"], -x[2]), reverse=True)
    
    if not substitutions:
        return f"No healthier substitutions found for {food_name}."
    
    result = f"### Healthier Substitutions for {nutrition['emoji']} {food_name}\n\n"
    
    for sub_name, data, _ in substitutions[:3]:  # Top 3
        result += f"- {data['emoji']} **{sub_name}** ({data['cal']} cal, Health Score: {data['health']}/10)\n"
    
    return result

def create_shareable_card(food_name, nutrition):
    """Create a shareable card for detected food"""
    health_rating, health_color = get_health_rating(nutrition["health_score"])
    
    card_html = f"""
    <div style='background: {COLORS["gradient_1"]}; 
                padding: 25px; border-radius: 15px; color: white; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.2); text-align: center;'>
        
        <h2 style='margin: 0 0 15px 0; font-size: 2em;'>
            {nutrition['emoji']} {food_name}
        </h2>
        
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 1.1em;'>
                <div><strong>🔥 Calories:</strong> {nutrition['calories']} kcal</div>
                <div><strong>💪 Protein:</strong> {nutrition['protein']}g</div>
                <div><strong>🍞 Carbs:</strong> {nutrition['carbs']}g</div>
                <div><strong>🥑 Fat:</strong> {nutrition['fat']}g</div>
            </div>
        </div>
        
        <div style='background: {health_color}; padding: 10px; border-radius: 10px; font-size: 1.1em; font-weight: bold; margin-bottom: 15px;'>
            {health_rating}
        </div>
        
        <div style='font-size: 0.9em; opacity: 0.9;'>
            Detected with Food Vision AI - Ultimate Pro
        </div>
    </div>
    """
    
    return card_html

def process_voice_command(voice_text):
    """Simulate voice command processing"""
    # In a real app, this would use speech-to-text and NLP
    # For demo, we'll extract keywords and return appropriate response
    
    text = voice_text.lower()
    
    if "calories" in text and "today" in text:
        daily_intake = get_daily_calorie_intake()
        return f"You've consumed {daily_intake} calories today."
    
    elif "water" in text:
        water_intake = get_water_intake_today()
        return f"You've had {water_intake} ml of water today."
    
    elif "bmi" in text:
        bmi, category, _ = calculate_bmi(USER_PROFILE["height_cm"], USER_PROFILE["weight_kg"])
        return f"Your BMI is {bmi}, which is considered {category}."
    
    elif "recommend" in text or "suggest" in text:
        return get_meal_recommendations()
    
    else:
        return "I'm not sure how to help with that. Try asking about calories, water intake, BMI, or recommendations."

# ==================== PREDICTION FUNCTION ====================
def predict_food(image):
    """
    Ultimate prediction function with all features
    """
    if image is None:
        return None, "❌ Please upload an image", "", "", ""
    
    start_time = time.time()
    
    try:
        # Convert to PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image = image.convert('RGB')
        
        # Preprocess
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top 5
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        # Format results
        results = {}
        predictions_list = []
        for prob, idx in zip(top5_prob.cpu().numpy(), top5_idx.cpu().numpy()):
            food_name = idx_to_label[int(idx)].replace('_', ' ').title()
            conf = float(prob)
            results[food_name] = conf
            predictions_list.append({"name": food_name, "confidence": conf})
        
        # Get top prediction details
        top_food = predictions_list[0]["name"]
        top_conf = predictions_list[0]["confidence"]
        
        # Get nutrition info
        nutrition = get_nutrition_info(top_food)
        health_rating, health_color = get_health_rating(nutrition["health_score"])
        fun_fact = get_fun_fact(top_food)
        
        # Get recipe info
        recipe = get_recipe_info(top_food)
        
        # Add to meal history
        meal = add_to_meal_history(top_food, nutrition, datetime.now())
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # ==================== FORMAT OUTPUTS ====================
        
        # 1. Confidence Message
        if top_conf > 0.8:
            conf_msg = f"""
### 🎯 **High Confidence Detection!**
The AI is **{top_conf*100:.1f}%** confident this is **{nutrition['emoji']} {top_food}**!

*Inference time: {inference_time:.0f}ms*
            """
        elif top_conf > 0.5:
            conf_msg = f"""
### ✅ **Good Detection!**
The AI is **{top_conf*100:.1f}%** confident this is **{nutrition['emoji']} {top_food}**.

*Inference time: {inference_time:.0f}ms*
            """
        else:
            conf_msg = f"""
### 🤔 **Uncertain Detection**
The AI is only **{top_conf*100:.1f}%** confident. Try:
- Better lighting
- Clearer photo
- Different angle

*Inference time: {inference_time:.0f}ms*
            """
        
        # 2. Nutrition Information
        nutrition_html = f"""
<div style='background: {COLORS["gradient_1"]}; 
            padding: 25px; border-radius: 15px; color: white; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
    
    <h2 style='margin: 0 0 20px 0; text-align: center; font-size: 2em;'>
        {nutrition['emoji']} {top_food}
    </h2>
    
    <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='margin: 0 0 15px 0; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px;'>
            📊 Nutrition Facts (per serving)
        </h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 1.1em;'>
            <div><strong>🔥 Calories:</strong> {nutrition['calories']} kcal</div>
            <div><strong>💪 Protein:</strong> {nutrition['protein']}g</div>
            <div><strong>🍞 Carbs:</strong> {nutrition['carbs']}g</div>
            <div><strong>🥑 Fat:</strong> {nutrition['fat']}g</div>
            <div><strong>🌾 Fiber:</strong> {nutrition['fiber']}g</div>
            <div><strong>⚡ Health Score:</strong> {nutrition['health_score']}/10</div>
            <div><strong>🍬 Sugar:</strong> {nutrition['sugar']}g</div>
            <div><strong>🧂 Sodium:</strong> {nutrition['sodium']}mg</div>
        </div>
    </div>
    
    <div style='background: {health_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 1.2em; font-weight: bold; margin-bottom: 20px;'>
        {health_rating}
    </div>
    
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; font-size: 1em;'>
        <strong>💡 Did you know?</strong><br>
        {fun_fact}
    </div>
</div>
        """
        
        # 3. Recipe Information
        recipe_html = f"""
<div style='background: {COLORS["gradient_2"]}; 
            padding: 25px; border-radius: 15px; color: white; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
    
    <h2 style='margin: 0 0 20px 0; text-align: center; font-size: 2em;'>
        🍳 {recipe['name']}
    </h2>
    
    <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; text-align: center;'>
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;'>
            <div style='font-size: 1.5em; font-weight: bold;'>⏱️</div>
            <div>{recipe['time']}</div>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;'>
            <div style='font-size: 1.5em; font-weight: bold;'>📊</div>
            <div>{recipe['difficulty']}</div>
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px;'>
            <div style='font-size: 1.5em; font-weight: bold;'>🍽️</div>
            <div>{recipe['servings']} servings</div>
        </div>
    </div>
    
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
        <h3 style='margin: 0 0 10px 0;'>🥘 Ingredients</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 0.9em;'>
        """
        
        for ingredient in recipe['ingredients']:
            recipe_html += f"<div>• {ingredient}</div>"
        
        recipe_html += f"""
        </div>
    </div>
    
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <h3 style='margin: 0 0 10px 0;'>📝 Instructions</h3>
        <ol style='padding-left: 20px; font-size: 0.9em;'>
        """
        
        for step in recipe['steps']:
            recipe_html += f"<li>{step}</li>"
        
        recipe_html += f"""
        </ol>
    </div>
    
    <div style='text-align: center; margin-top: 15px; font-size: 0.9em; opacity: 0.9;'>
        🔥 {recipe['calories']} calories per serving
    </div>
</div>
        """
        
        # 4. Alternative Predictions
        alternatives = "### 🔄 Other Possibilities:\n"
        for i, pred in enumerate(predictions_list[1:5], 1):
            bar_width = int(pred['confidence'] * 100)
            alternatives += f"{i}. **{pred['name']}**: {pred['confidence']*100:.1f}% {'▰' * (bar_width//10)}{'▱' * (10-bar_width//10)}\n"
        
        # 5. Recommendations
        if nutrition['health_score'] >= 8:
            recommendations = """
### 🌟 Excellent Choice!
This is a nutritious option. Great for regular meals!

**Tips:**
- ✅ High in protein/nutrients
- ✅ Good for muscle building
- ✅ Supports overall health
            """
        elif nutrition['health_score'] >= 6:
            recommendations = """
### 👍 Good Choice!
A balanced option. Enjoy as part of a varied diet.

**Tips:**
- ⚖️ Moderate in calories
- ⚖️ Balance with vegetables
- ⚖️ Watch portion sizes
            """
        else:
            recommendations = """
### ⚠️ Occasional Treat
Enjoy in moderation. Balance with healthier options.

**Tips:**
- 🥗 Pair with salad or vegetables
- 💧 Stay hydrated
- 🏃 Consider extra exercise
- 🍎 Balance with fruits later
            """
        
        # 6. Daily Progress
        daily_intake = get_daily_calorie_intake()
        remaining = USER_PROFILE["daily_calorie_goal"] - daily_intake
        progress_percent = min(100, (daily_intake / USER_PROFILE["daily_calorie_goal"]) * 100)
        
        daily_progress = f"""
### 📊 Today's Progress
- **Calories Consumed:** {daily_intake} / {USER_PROFILE["daily_calorie_goal"]}
- **Remaining:** {remaining} calories
- **Progress:** {progress_percent:.1f}%

{'🎉 Goal reached!' if remaining <= 0 else '💪 Keep going!'}
        """
        
        # 7. Shareable Card
        shareable_card = create_shareable_card(top_food, nutrition)
        
        # 8. Food Substitutions
        substitutions = get_food_substitutions(top_food)
        
        # Combine everything
        full_output = f"""
{conf_msg}

---

{alternatives}

---

{recommendations}

---

{daily_progress}

---

{substitutions}

---

<div style='text-align: center; font-size: 0.9em; color: {COLORS["muted"]}; padding: 10px;'>
    <strong>📅 Detected on:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
</div>
        """
        
        return results, nutrition_html, recipe_html, full_output, shareable_card
    
    except Exception as e:
        error_msg = f"""
### ❌ Error Occurred

**Error:** {str(e)}

**Possible solutions:**
- Make sure the image is clear
- Try a different photo
- Check image format (JPG/PNG)
        """
        return None, f"❌ Error: {str(e)}", "", error_msg, ""

# ==================== MOBILE-FIRST CSS ====================
MOBILE_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global styles */
* {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* Mobile-first responsive design */
html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow-x: hidden;
}}

body {{
    background-color: {COLORS["light"]};
    color: {COLORS["dark"]};
    line-height: 1.6;
}}

/* Container */
.gradio-container {{
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 0 !important;
}}

/* Header styles */
#title {{
    text-align: center;
    font-size: clamp(2rem, 8vw, 3.5rem) !important;
    font-weight: 900 !important;
    background: {COLORS["gradient_primary"]};
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0.5rem 0 1rem 0 !important;
    text-shadow: none !important;
    animation: gradient 3s ease infinite;
    padding: 0 1rem;
    background-size: 200% 200%;
}}

#subtitle {{
    text-align: center;
    font-size: clamp(1rem, 4vw, 1.3rem) !important;
    color: {COLORS["muted"]} !important;
    margin-bottom: 1.5rem !important;
    font-weight: 400 !important;
    padding: 0 1rem;
    line-height: 1.5;
}}

/* Stats box */
.stats-box {{
    background: {COLORS["gradient_1"]};
    padding: 1.5rem !important;
    border-radius: 1rem !important;
    color: white !important;
    text-align: center !important;
    margin: 1rem 0 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
}}

/* Feature cards */
.feature-card {{
    background: white !important;
    border-radius: 1rem !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    transition: transform 0.3s ease !important;
}}

.feature-card:hover {{
    transform: translateY(-5px) !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
}}

/* Buttons */
button {{
    border-radius: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
    touch-action: manipulation !important;
    -webkit-tap-highlight-color: transparent !important;
}}

button:hover {{
    transform: scale(1.05) !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}}

button:active {{
    transform: scale(0.98) !important;
}}

/* Form elements */
input, textarea, select {{
    font-size: 16px !important;
    touch-action: manipulation !important;
    min-height: 44px !important;
    border-radius: 0.5rem !important;
    border: 1px solid {COLORS["border"]} !important;
    padding: 0.5rem 0.75rem !important;
    background-color: {COLORS["white"]} !important;
    color: {COLORS["dark"]} !important;
}}

/* Tabs */
.tab-nav {{
    background: {COLORS["gradient_1"]} !important;
    border-radius: 0.75rem 0.75rem 0 0 !important;
}}

.tab-nav button {{
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
    min-height: 44px !important;
}}

.tab-nav button.selected {{
    background: rgba(255,255,255,0.2) !important;
}}

/* Image input */
#image-input {{
    border-radius: 1rem !important;
    border: 2px dashed {COLORS["border_dark"]} !important;
    transition: all 0.3s ease !important;
    background: {COLORS["gradient_soft"]} !important;
    min-height: 300px !important;
}}

#image-input:hover {{
    border-color: {COLORS["primary"]} !important;
    background: {COLORS["gradient_soft"]} !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}}

#image-input img {{
    border-radius: 0.75rem !important;
    max-height: 300px !important;
    object-fit: contain !important;
}}

/* Progress bar */
.progress-container {{
    width: 100% !important;
    background-color: {COLORS["border"]} !important;
    border-radius: 0.5rem !important;
    margin: 0.5rem 0 !important;
    height: 1rem !important;
    overflow: hidden !important;
}}

.progress-bar {{
    height: 100% !important;
    border-radius: 0.5rem !important;
    background: {COLORS["gradient_1"]} !important;
    transition: width 0.5s ease !important;
    box-shadow: 0 2px 4px rgba(79, 70, 229, 0.3) !important;
}}

/* Water container */
.water-container {{
    position: relative !important;
    width: 80px !important;
    height: 160px !important;
    background-color: {COLORS["border"]} !important;
    border-radius: 40px 40px 0 0 !important;
    margin: 0 auto !important;
    overflow: hidden !important;
    border: 2px solid {COLORS["border_dark"]} !important;
}}

.water-fill {{
    position: absolute !important;
    bottom: 0 !important;
    width: 100% !important;
    background: {COLORS["gradient_3"]} !important;
    transition: height 0.5s ease !important;
    box-shadow: inset 0 -2px 10px rgba(59, 130, 246, 0.3) !important;
}}

/* Shopping list */
.shopping-item {{
    display: flex !important;
    align-items: center !important;
    padding: 0.75rem !important;
    border-bottom: 1px solid {COLORS["border"]} !important;
    background-color: {COLORS["white"]} !important;
    transition: background-color 0.2s ease !important;
}}

.shopping-item:hover {{
    background-color: {COLORS["light"]} !important;
}}

.shopping-item.checked {{
    opacity: 0.6 !important;
    text-decoration: line-through !important;
}}

.shopping-checkbox {{
    margin-right: 0.5rem !important;
}}

/* BMI result */
.bmi-result {{
    padding: 1rem !important;
    border-radius: 0.75rem !important;
    text-align: center !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1rem !important;
}}

.bmi-underweight {{
    background: {COLORS["gradient_3"]} !important;
}}

.bmi-normal {{
    background: {COLORS["gradient_4"]} !important;
}}

.bmi-overweight {{
    background: {COLORS["gradient_5"]} !important;
}}

.bmi-obese {{
    background: {COLORS["gradient_6"]} !important;
}}

/* Animations */
@keyframes gradient {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

@keyframes pulse {{
    0% {{transform: scale(1);}}
    50% {{transform: scale(1.05);}}
    100% {{transform: scale(1);}}
}}

.pulse {{
    animation: pulse 2s infinite;
}}

/* Mobile-specific optimizations */
@media (max-width: 768px) {{
    .gradio-container {{
        padding: 0 0.5rem !important;
    }}
    
    .stats-box {{
        padding: 1rem !important;
    }}
    
    .feature-card {{
        padding: 1rem !important;
    }}
    
    button {{
        width: 100% !important;
        margin: 0.25rem 0 !important;
    }}
    
    .tab-nav button {{
        font-size: 0.9rem !important;
        padding: 0.5rem 0.75rem !important;
    }}
    
    #image-input {{
        min-height: 250px !important;
    }}
    
    .water-container {{
        width: 60px !important;
        height: 120px !important;
    }}
}}

/* Very small screens */
@media (max-width: 480px) {{
    .gradio-container {{
        padding: 0 0.25rem !important;
    }}
    
    #title {{
        font-size: 1.8rem !important;
    }}
    
    #subtitle {{
        font-size: 1rem !important;
    }}
    
    .stats-box {{
        padding: 0.75rem !important;
    }}
    
    .feature-card {{
        padding: 0.75rem !important;
    }}
    
    button {{
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }}
    
    #image-input {{
        min-height: 200px !important;
    }}
    
    .water-container {{
        width: 50px !important;
        height: 100px !important;
    }}
}}

/* Landscape mobile */
@media (max-width: 896px) and (orientation: landscape) {{
    .gradio-container {{
        padding: 0 0.5rem !important;
    }}
    
    #title {{
        font-size: 2rem !important;
    }}
    
    .stats-box {{
        padding: 1rem !important;
    }}
    
    .feature-card {{
        padding: 1rem !important;
    }}
    
    #image-input {{
        min-height: 200px !important;
    }}
}}

/* Touch feedback */
.touch-feedback {{
    position: relative !important;
    overflow: hidden !important;
}}

.touch-feedback::before {{
    content: '' !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    width: 0 !important;
    height: 0 !important;
    border-radius: 50% !important;
    background: rgba(79, 70, 229, 0.2) !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.3s ease, height 0.3s ease !important;
    pointer-events: none !important;
}}

.touch-feedback:active::before {{
    width: 100px !important;
    height: 100px !important;
}}

/* GPU acceleration for performance */
.gpu-accelerated {{
    transform: translateZ(0) !important;
    will-change: transform !important;
    backface-visibility: hidden !important;
    -webkit-backface-visibility: hidden !important;
}}

/* Lazy loading */
.lazy-load {{
    opacity: 0 !important;
    transform: translateY(20px) !important;
    transition: opacity 0.3s ease, transform 0.3s ease !important;
}}

.lazy-load.loaded {{
    opacity: 1 !important;
    transform: translateY(0) !important;
}}

/* Network optimization */
.network-optimized {{
    image-rendering: -webkit-optimize-contrast !important;
    image-rendering: crisp-edges !important;
}}

/* Performance monitoring */
.performance-monitor {{
    position: fixed !important;
    bottom: 10px !important;
    right: 10px !important;
    background: rgba(0,0,0,0.7) !important;
    color: white !important;
    padding: 5px 10px !important;
    border-radius: 5px !important;
    font-size: 0.8rem !important;
    z-index: 1000 !important;
    display: none !important;
}}
"""

# ==================== GRADIO INTERFACE ====================

# Build the interface with mobile-first design
with gr.Blocks(
    css=MOBILE_CSS, 
    theme=gr.themes.Soft(
        primary_hue="indigo", 
        secondary_hue="pink",
        font=[gr.themes.GoogleFont("Inter"), gr.themes.GoogleFont("Roboto")]
    ),
    title="🍕 Food Vision AI - Ultimate Pro",
    analytics_enabled=False,
    head=f"""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="{COLORS["primary"]}">
    <link rel="manifest" href="data:application/json;base64,eyJuYW1lIjoiRm9vZCBWaXNpb24gQUkiLCJzaG9ydF9uYW1lIjoiRm9vZEFJIn0=">
    
    <!-- Performance optimizations -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    
    <!-- Mobile-specific meta tags -->
    <meta name="format-detection" content="telephone=no">
    <meta name="msapplication-tap-highlight" content="no">
    
    <!-- PWA manifest -->
    <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='{COLORS["primary"]}'/><text x='50' y='55' font-family='Arial' font-size='40' fill='white' text-anchor='middle'>🍕</text></svg>">
    
    <!-- Performance monitoring -->
    <script>
    // Performance monitoring for mobile
    if ('performance' in window) {{
        window.addEventListener('load', function() {{
            setTimeout(function() {{
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load time:', perfData.loadEventEnd - perfData.fetchStart, 'ms');
            }}, 0);
        }});
    }}
    
    // Network optimization
    if ('connection' in navigator) {{
        const connection = navigator.connection;
        if (connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g') {{
            document.documentElement.classList.add('network-optimized');
        }}
    }}
    
    // Touch optimization
    if ('ontouchstart' in window) {{
        document.documentElement.classList.add('touch-device');
    }}
    </script>
    """
) as demo:
    
    # Mobile-optimized JavaScript
    gr.HTML(f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        // Add touch event listeners for mobile feedback
        const touchElements = document.querySelectorAll('button, .gradio-button, .tab-nav, .feature-card');
        touchElements.forEach(element => {{
            element.classList.add('touch-feedback');
            
            element.addEventListener('touchstart', function(e) {{
                this.style.transform = 'scale(0.98)';
                this.style.transition = 'transform 0.1s ease';
            }}, {{ passive: true }});

            element.addEventListener('touchend', function(e) {{
                this.style.transform = 'scale(1)';
                setTimeout(() => {{
                    this.style.transition = '';
                }}, 100);
            }}, {{ passive: true }});

            element.addEventListener('touchcancel', function(e) {{
                this.style.transform = 'scale(1)';
                this.style.transition = '';
            }}, {{ passive: true }});
        }});

        // Mobile image input optimization
        const imageInput = document.querySelector('input[type="file"]');
        if (imageInput) {{
            imageInput.addEventListener('change', async function(e) {{
                const file = e.target.files[0];
                if (file && window.innerWidth <= 768) {{
                    // Show mobile compression indicator
                    const compressionIndicator = document.createElement('div');
                    compressionIndicator.className = 'mobile-compression-indicator';
                    compressionIndicator.innerHTML = '<div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: {COLORS["primary"]}; color: white; padding: 1rem; border-radius: 8px; z-index: 10000; text-align: center;"><div>📸 Optimizing for mobile...</div><div style="font-size: 0.8rem; margin-top: 0.5rem;">Compressing image for faster analysis</div></div>';
                    document.body.appendChild(compressionIndicator);

                    try {{
                        // Process image with mobile optimizations
                        const optimizedFile = await handleMobileImageUpload(file);

                        // Replace the file input with optimized file
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(optimizedFile);
                        e.target.files = dataTransfer.files;

                        // Remove compression indicator
                        setTimeout(() => {{
                            if (compressionIndicator.parentNode) {{
                                compressionIndicator.parentNode.removeChild(compressionIndicator);
                            }}
                        }}, 500);

                    }} catch (error) {{
                        console.warn('Mobile optimization failed, using original file:', error);
                        if (compressionIndicator.parentNode) {{
                            compressionIndicator.parentNode.removeChild(compressionIndicator);
                        }}
                    }}
                }}
            }});
        }}

        // Mobile performance monitoring
        if (window.innerWidth <= 768) {{
            // Monitor touch response time
            let touchStartTime;

            document.addEventListener('touchstart', function(e) {{
                touchStartTime = performance.now();
            }}, {{ passive: true }});

            document.addEventListener('touchend', function(e) {{
                if (touchStartTime) {{
                    const responseTime = performance.now() - touchStartTime;
                    if (responseTime > 100) {{
                        console.warn('Slow touch response: ' + responseTime.toFixed(2) + 'ms');
                    }}
                }}
            }}, {{ passive: true }});

            // Optimize scrolling performance
            let ticking = false;
            
            document.addEventListener('scroll', function() {{
                if (!ticking) {{
                    window.requestAnimationFrame(function() {{
                        // Add scroll-based optimizations here
                        ticking = false;
                    }});
                    ticking = true;
                }}
            }}, {{ passive: true }});
        }}

        // Network optimization for mobile data
        if ('connection' in navigator) {{
            const connection = navigator.connection;
            const isSlowConnection = connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g';

            if (isSlowConnection) {{
                const slowIndicator = document.createElement('div');
                slowIndicator.className = 'mobile-network-indicator';
                slowIndicator.innerHTML = '<div style="position: fixed; top: 0; left: 0; width: 100%; background: {COLORS["warning"]}; color: white; padding: 0.5rem; text-align: center; z-index: 10000;"><b>Slow Network Detected.</b> Optimizing for speed...</div>';
                document.body.appendChild(slowIndicator);
                
                // Auto-hide after 5 seconds
                setTimeout(() => {{
                    if (slowIndicator.parentNode) {{
                        slowIndicator.parentNode.removeChild(slowIndicator);
                    }}
                }}, 5000);
            }}
        }}

        // Add GPU acceleration to elements
        document.querySelectorAll('.feature-card, .stat-item, .accordion-header').forEach(function(el) {{
            el.classList.add('gpu-accelerated');
        }});

        // Add lazy loading to images
        document.querySelectorAll('img').forEach(function(img) {{
            img.classList.add('lazy-load');
            
            // Simple intersection observer for lazy loading
            if ('IntersectionObserver' in window) {{
                const imageObserver = new IntersectionObserver(function(entries, observer) {{
                    entries.forEach(function(entry) {{
                        if (entry.isIntersecting) {{
                            img.classList.add('loaded');
                            observer.unobserve(img);
                        }}
                    }});
                }});
                
                imageObserver.observe(img);
            }} else {{
                // Fallback for older browsers
                img.classList.add('loaded');
            }}
        }});

        // Handle mobile image upload optimization
        window.handleMobileImageUpload = async function(file) {{
            return new Promise((resolve, reject) => {{
                const reader = new FileReader();
                
                reader.onload = function(e) {{
                    const img = new Image();
                    
                    img.onload = function() {{
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        
                        // Calculate new dimensions
                        let {{ width, height }} = img;
                        const maxWidth = 1280;
                        const maxHeight = 1280;
                        
                        if (width > maxWidth || height > maxHeight) {{
                            const ratio = Math.min(maxWidth / width, maxHeight / height);
                            width *= ratio;
                            height *= ratio;
                        }}
                        
                        canvas.width = width;
                        canvas.height = height;
                        
                        // Draw and compress
                        ctx.drawImage(img, 0, 0, width, height);
                        
                        canvas.toBlob(function(blob) {{
                            resolve(new File([blob], file.name, {{
                                type: 'image/jpeg',
                                lastModified: Date.now()
                            }}));
                        }}, 'image/jpeg', 0.8);
                    }};
                    
                    img.src = e.target.result;
                }};
                
                reader.readAsDataURL(file);
            }});
        }};

        // Performance monitoring
        if (window.innerWidth <= 768) {{
            const perfMonitor = document.createElement('div');
            perfMonitor.className = 'performance-monitor';
            document.body.appendChild(perfMonitor);
            
            // Show performance monitor in development
            if (window.location.hostname === 'localhost') {{
                perfMonitor.style.display = 'block';
                
                let frameCount = 0;
                let lastTime = performance.now();
                
                function updatePerfMonitor() {{
                    frameCount++;
                    const currentTime = performance.now();
                    
                    if (currentTime - lastTime >= 1000) {{
                        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                        perfMonitor.textContent = `FPS: ${{fps}}`;
                        
                        frameCount = 0;
                        lastTime = currentTime;
                    }}
                    
                    requestAnimationFrame(updatePerfMonitor);
                }}
                
                requestAnimationFrame(updatePerfMonitor);
            }}
        }}
    }});
    </script>
    """)
    
    # Header with mobile optimization
    gr.HTML(f"""
    <div style='text-align: center; padding: 1rem 0.5rem; background: linear-gradient(135deg, {COLORS["light"]} 0%, {COLORS["light_dark"]} 100%); border-radius: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
        <h1 id='title' class='floating-element'>🍕 FOOD VISION AI</h1>
        <p id='subtitle' style='margin: 0.5rem 0; font-size: 1rem; color: {COLORS["muted"]}; max-width: 700px; margin-left: auto; margin-right: auto; line-height: 1.5;'>
            🚀 The world's most advanced AI-powered food recognition system | 
            Elite detection with 101+ food categories | 
            Real-time nutrition analysis | 
            Personalized health insights
        </p>
        <div style='margin-top: 0.5rem; font-size: 0.8rem; color: {COLORS["muted"]};'>
            <span>🛡️ Powered by EfficientNet-B0 | </span>
            <span>⚡ 81.1% Top-1 Accuracy | </span>
            <span>🎯 <100ms Inference | </span>
            <span>🔒 Privacy-First Design</span>
        </div>
    </div>
    """)
    
    # Stats Box with mobile optimization
    gr.HTML(f"""
    <div class='stats-box gpu-accelerated'>
        <h2 style='margin: 0 0 0.75rem 0; font-size: 1.5rem; font-weight: 700; text-align: center;'>🎯 Elite Performance</h2>
        <p style='margin: 0 0 1rem 0; font-size: 0.9rem; opacity: 0.95; text-align: center;'>
            State-of-the-art computer vision trained on 101,000+ professional food images
        </p>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin: 1rem 0;'>
            <div style='text-align: center; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem;'>
                <div style='font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;'>81.10%</div>
                <div style='font-size: 0.8rem; opacity: 0.9;'>Top-1 Accuracy</div>
            </div>
            <div style='text-align: center; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem;'>
                <div style='font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;'>95.64%</div>
                <div style='font-size: 0.8rem; opacity: 0.9;'>Top-5 Accuracy</div>
            </div>
            <div style='text-align: center; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem;'>
                <div style='font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;'>101+</div>
                <div style='font-size: 0.8rem; opacity: 0.9;'>Food Categories</div>
            </div>
            <div style='text-align: center; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem;'>
                <div style='font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;'><100ms</div>
                <div style='font-size: 0.8rem; opacity: 0.9;'>Response Time</div>
            </div>
        </div>
        
        <div style='margin-top: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem; font-size: 0.8rem; text-align: center;'>
            🧠 Architecture: EfficientNet-B0 (Google Research) | 
            🔧 Framework: PyTorch 2.0 + TIMM | 
            📚 Dataset: Food-101 (ETH Zurich) | 
            🎓 Training: Transfer Learning + Fine-tuning
        </div>
    </div>
    """)
    
    # Main Interface with Tabs - Mobile-First Layout
    with gr.Tabs() as tabs:
        with gr.Tab("📸 Food Detection", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                # Image Input with Mobile Optimization
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📸 Upload Food Image")
                    image_input = gr.Image(
                        label="",
                        type="pil",
                        height=300,
                        elem_id="image-input",
                        sources=["upload", "webcam", "clipboard"],
                        elem_classes="mobile-camera-optimized"
                    )
                    
                    # Mobile-Optimized Buttons
                    with gr.Row(elem_classes="mobile-full-width"):
                        clear_btn = gr.ClearButton(
                            [image_input], 
                            value="🗑️ Clear", 
                            variant="secondary",
                            elem_classes="touch-feedback"
                        )
                        submit_btn = gr.Button(
                            "🔍 ANALYZE FOOD",
                            variant="primary",
                            size="lg",
                            elem_classes="pulse-animation touch-feedback gpu-accelerated"
                        )
                
                # Mobile-Optimized Tips
                with gr.Accordion("💡 Pro Tips", open=False, elem_classes="feature-card"):
                    gr.HTML(f"""
                    <div style='line-height: 1.6;'>
                        <div style='display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.75rem;'>
                            <span style='font-size: 1.2rem;'>📷</span>
                            <div><strong>Good lighting</strong> - Natural light works best</div>
                        </div>
                        <div style='display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.75rem;'>
                            <span style='font-size: 1.2rem;'>🎯</span>
                            <div><strong>Center the food</strong> - Fill most of the frame</div>
                        </div>
                        <div style='display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.75rem;'>
                            <span style='font-size: 1.2rem;'>🚫</span>
                            <div><strong>Avoid filters</strong> - Use original photos</div>
                        </div>
                        <div style='display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.75rem;'>
                            <span style='font-size: 1.2rem;'>🍽️</span>
                            <div><strong>One item at a time</strong> - Single food works better</div>
                        </div>
                        <div style='display: flex; align-items: flex-start; gap: 0.5rem;'>
                            <span style='font-size: 1.2rem;'>📐</span>
                            <div><strong>Try different angles</strong> - Top-down or 45° angle</div>
                        </div>
                    </div>
                    """)
                
                # Results Section - Mobile Optimized
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🎯 AI Results")
                    
                    # Predictions - Mobile Optimized
                    predictions_output = gr.Label(
                        label="",
                        num_top_classes=3,  # Reduced for mobile
                        elem_id="predictions",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Nutrition Info - Mobile Optimized
                    nutrition_output = gr.HTML(
                        label="📊 Nutrition",
                        elem_id="elite-nutrition",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Recipe Info - Mobile Optimized
                    recipe_output = gr.HTML(
                        label="🍳 Recipe",
                        elem_id="elite-recipe",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Detailed Analysis - Mobile Optimized
                    analysis_output = gr.Markdown(
                        label="📋 Analysis",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Shareable Card - Mobile Optimized
                    shareable_output = gr.HTML(
                        label="📤 Share",
                        elem_id="shareable-card",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("📊 Meal Tracking", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📈 Your Daily Progress")
                    
                    # Daily progress display - Mobile Optimized
                    daily_progress_display = gr.Markdown(
                        value="Upload a food image to start tracking your meals!",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Progress bar - Mobile Optimized
                    progress_bar_html = gr.HTML(
                        value=f"<div class='progress-container'><div class='progress-bar' style='width: 0%; background: {COLORS['gradient_1']};'></div></div>",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Weekly nutrition chart - Mobile Optimized
                    weekly_chart = gr.Plot(
                        label="Weekly Nutrition",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Export button - Mobile Optimized
                    export_btn = gr.Button(
                        "📥 Export Report",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                    
                    export_file = gr.File(
                        label="Download",
                        visible=False,
                        elem_classes="mobile-full-width"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📝 Meal History")
                    
                    # Meal history display - Mobile Optimized
                    meal_history_display = gr.DataFrame(
                        headers=["Food", "Calories", "Time"],
                        datatype=["str", "number", "str"],
                        value=[],
                        label="Recent Meals",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Refresh button - Mobile Optimized
                    refresh_btn = gr.Button(
                        "🔄 Refresh",
                        variant="secondary",
                        elem_classes="touch-feedback"
                    )
                    
                    # Recommendations - Mobile Optimized
                    recommendations_display = gr.Markdown(
                        value="Upload a food image to get personalized recommendations!",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("💧 Water & Exercise", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 💧 Water Intake")
                    
                    # Water intake visualization - Mobile Optimized
                    water_visual = gr.HTML(
                        value=f"""
                        <div class='water-container'>
                            <div class='water-fill' style='height: 0%; background: linear-gradient(180deg, {COLORS["info"]} 0%, {COLORS["primary"]} 100%);'></div>
                        </div>
                        <p style='text-align: center; margin-top: 10px;'>0 ml / 2000 ml</p>
                        """,
                        elem_classes="mobile-full-width"
                    )
                    
                    # Water intake buttons - Mobile Optimized
                    with gr.Row(elem_classes="mobile-full-width"):
                        water_250_btn = gr.Button("250 ml", variant="secondary", elem_classes="touch-feedback")
                        water_500_btn = gr.Button("500 ml", variant="secondary", elem_classes="touch-feedback")
                        water_1000_btn = gr.Button("1000 ml", variant="secondary", elem_classes="touch-feedback")
                    
                    # Custom water input - Mobile Optimized
                    with gr.Row(elem_classes="mobile-full-width"):
                        custom_water_input = gr.Number(
                            label="Custom Amount (ml)",
                            value=250,
                            minimum=50,
                            maximum=1000,
                            elem_classes="mobile-full-width"
                        )
                        add_water_btn = gr.Button("Add", variant="primary", elem_classes="touch-feedback")
                    
                    # Water goal setting - Mobile Optimized
                    water_goal_input = gr.Number(
                        label="Daily Goal (ml)",
                        value=USER_PROFILE["water_goal_ml"],
                        minimum=1000,
                        maximum=5000,
                        elem_classes="mobile-full-width"
                    )
                    set_water_goal_btn = gr.Button("Set Goal", variant="secondary", elem_classes="touch-feedback")
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🏃 Exercise Tracking")
                    
                    # Exercise input - Mobile Optimized
                    exercise_type = gr.Dropdown(
                        label="Activity",
                        choices=[
                            "Walking", "Running", "Cycling", "Swimming", 
                            "Weight Training", "Yoga", "Dancing", "Sports"
                        ],
                        value="Walking",
                        elem_classes="mobile-full-width"
                    )
                    
                    with gr.Row(elem_classes="mobile-full-width"):
                        exercise_duration = gr.Number(
                            label="Duration (min)",
                            value=30,
                            minimum=5,
                            maximum=300,
                            elem_classes="mobile-full-width"
                        )
                        exercise_calories = gr.Number(
                            label="Calories",
                            value=150,
                            minimum=10,
                            maximum=1000,
                            elem_classes="mobile-full-width"
                        )
                    
                    add_exercise_btn = gr.Button("Add Exercise", variant="primary", elem_classes="touch-feedback")
                    
                    # Today's exercise list - Mobile Optimized
                    exercise_list = gr.DataFrame(
                        headers=["Activity", "Duration", "Calories"],
                        datatype=["str", "number", "number"],
                        value=[],
                        label="Today's Exercise",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("🍳 Recipe Book", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🔍 Search Recipes")
                    
                    # Recipe search - Mobile Optimized
                    recipe_search = gr.Textbox(
                        label="Search",
                        placeholder="e.g., Pizza, Sushi",
                        elem_classes="mobile-full-width"
                    )
                    
                    search_btn = gr.Button(
                        "🔍 Search",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                    
                    # Recipe categories - Mobile Optimized
                    gr.Markdown("### 📂 Categories")
                    
                    with gr.Row(elem_classes="mobile-full-width"):
                        breakfast_btn = gr.Button("🍳 Breakfast", variant="secondary", elem_classes="touch-feedback")
                        lunch_btn = gr.Button("🥗 Lunch", variant="secondary", elem_classes="touch-feedback")
                        dinner_btn = gr.Button("🍽️ Dinner", variant="secondary", elem_classes="touch-feedback")
                        dessert_btn = gr.Button("🍰 Dessert", variant="secondary", elem_classes="touch-feedback")
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📖 Recipe Details")
                    
                    # Recipe display - Mobile Optimized
                    recipe_display = gr.HTML(
                        value=f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Search for a recipe</div>",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("⚖️ Food Comparison", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🍎 Compare Foods")
                    
                    # Food selection - Mobile Optimized
                    food1_dropdown = gr.Dropdown(
                        label="First Food",
                        choices=[v.replace('_', ' ').title() for v in idx_to_label.values()],
                        value="Pizza",
                        elem_classes="mobile-full-width"
                    )
                    
                    food2_dropdown = gr.Dropdown(
                        label="Second Food",
                        choices=[v.replace('_', ' ').title() for v in idx_to_label.values()],
                        value="Sushi",
                        elem_classes="mobile-full-width"
                    )
                    
                    compare_btn = gr.Button(
                        "⚖️ Compare",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    # Comparison chart - Mobile Optimized
                    comparison_chart = gr.Plot(
                        label="Nutrition Comparison",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Comparison text - Mobile Optimized
                    comparison_text = gr.Markdown(
                        value="Select two foods to compare",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("📋 Meal Planner", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🗓️ Generate Plan")
                    
                    # Meal plan inputs - Mobile Optimized
                    meal_type = gr.Dropdown(
                        label="Meal Type",
                        choices=["Breakfast", "Lunch", "Dinner", "Snack"],
                        value="Breakfast",
                        elem_classes="mobile-full-width"
                    )
                    
                    calorie_target = gr.Number(
                        label="Calorie Target",
                        value=500,
                        minimum=100,
                        maximum=1000,
                        elem_classes="mobile-full-width"
                    )
                    
                    generate_plan_btn = gr.Button(
                        "📋 Generate",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    # Meal plan output - Mobile Optimized
                    meal_plan_output = gr.Markdown(
                        value="Select meal type and calorie target",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("🛒 Shopping List", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### ➕ Add Items")
                    
                    # Shopping list input - Mobile Optimized
                    item_name = gr.Textbox(
                        label="Item",
                        placeholder="e.g., Milk, Eggs",
                        elem_classes="mobile-full-width"
                    )
                    
                    item_quantity = gr.Textbox(
                        label="Quantity",
                        placeholder="e.g., 1 gallon",
                        elem_classes="mobile-full-width"
                    )
                    
                    add_item_btn = gr.Button(
                        "➕ Add",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📝 Shopping List")
                    
                    # Shopping list display - Mobile Optimized
                    shopping_list_display = gr.HTML(
                        value=f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Your list is empty</div>",
                        elem_classes="mobile-full-width"
                    )
                    
                    # Shopping list controls - Mobile Optimized
                    with gr.Row(elem_classes="mobile-full-width"):
                        refresh_list_btn = gr.Button(
                            "🔄 Refresh",
                            variant="secondary",
                            elem_classes="touch-feedback"
                        )
                        clear_list_btn = gr.Button(
                            "🗑️ Clear All",
                            variant="secondary",
                            elem_classes="touch-feedback"
                        )
        
        with gr.Tab("📊 Health Metrics", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📏 BMI Calculator")
                    
                    # BMI inputs - Mobile Optimized
                    height_input = gr.Number(
                        label="Height (cm)",
                        value=USER_PROFILE["height_cm"],
                        minimum=100,
                        maximum=250,
                        elem_classes="mobile-full-width"
                    )
                    
                    weight_input = gr.Number(
                        label="Weight (kg)",
                        value=USER_PROFILE["weight_kg"],
                        minimum=30,
                        maximum=200,
                        elem_classes="mobile-full-width"
                    )
                    
                    calculate_bmi_btn = gr.Button(
                        "📊 Calculate",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                    
                    # BMI result - Mobile Optimized
                    bmi_result = gr.HTML(
                        value=f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Enter your details</div>",
                        elem_classes="mobile-full-width"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 🔥 Calorie Calculator")
                    
                    # Calorie calculator inputs - Mobile Optimized
                    age_input = gr.Number(
                        label="Age",
                        value=30,
                        minimum=15,
                        maximum=100,
                        elem_classes="mobile-full-width"
                    )
                    
                    gender_input = gr.Radio(
                        label="Gender",
                        choices=["Male", "Female"],
                        value="Male",
                        elem_classes="mobile-full-width"
                    )
                    
                    activity_input = gr.Dropdown(
                        label="Activity Level",
                        choices=[
                            "Sedentary",
                            "Light",
                            "Moderate",
                            "Active",
                            "Very Active"
                        ],
                        value="Moderate",
                        elem_classes="mobile-full-width"
                    )
                    
                    calculate_calories_btn = gr.Button(
                        "🔥 Calculate",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                    
                    # Calorie result - Mobile Optimized
                    calorie_result = gr.HTML(
                        value=f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Enter your details</div>",
                        elem_classes="mobile-full-width"
                    )
        
        with gr.Tab("⚙️ Settings", elem_classes="tab-nav"):
            # Mobile-First: Single Column Layout
            with gr.Column():
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 👤 Profile")
                    
                    # User settings - Mobile Optimized
                    name_input = gr.Textbox(
                        label="Name",
                        value=USER_PROFILE["name"],
                        elem_classes="mobile-full-width"
                    )
                    
                    email_input = gr.Textbox(
                        label="Email",
                        value=USER_PROFILE["email"],
                        elem_classes="mobile-full-width"
                    )
                    
                    calorie_goal_input = gr.Number(
                        label="Daily Calorie Goal",
                        value=USER_PROFILE["daily_calorie_goal"],
                        minimum=1000,
                        maximum=5000,
                        elem_classes="mobile-full-width"
                    )
                    
                    dietary_preferences = gr.CheckboxGroup(
                        label="Dietary Preferences",
                        choices=["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Low-Carb"],
                        value=USER_PROFILE["dietary_preferences"],
                        elem_classes="mobile-full-width"
                    )
                    
                    allergies = gr.CheckboxGroup(
                        label="Allergies",
                        choices=["Nuts", "Dairy", "Gluten", "Eggs", "Soy", "Shellfish"],
                        value=USER_PROFILE["allergies"],
                        elem_classes="mobile-full-width"
                    )
                    
                    # Placeholder for status messages
                    status_output = gr.Markdown(
                        value="",
                        visible=False,
                        elem_classes="mobile-full-width"
                    )

                    save_profile_btn = gr.Button(
                        "💾 Save",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                
                with gr.Group(elem_classes="feature-card"):
                    gr.Markdown("### 📱 App Settings")
                    
                    # App settings - Mobile Optimized
                    theme_selector = gr.Dropdown(
                        label="Theme",
                        choices=["Light", "Dark", "Auto"],
                        value="Light",
                        elem_classes="mobile-full-width"
                    )
                    
                    notifications = gr.Checkbox(
                        label="Notifications",
                        value=True,
                        elem_classes="mobile-full-width"
                    )
                    
                    auto_save = gr.Checkbox(
                        label="Auto-save",
                        value=True,
                        elem_classes="mobile-full-width"
                    )
                    
                    save_settings_btn = gr.Button(
                        "💾 Save",
                        variant="primary",
                        elem_classes="touch-feedback"
                    )
                    
                    gr.Markdown("### 📊 Data & Privacy")
                    
                    clear_data_btn = gr.Button(
                        "🗑️ Clear Data",
                        variant="secondary",
                        elem_classes="touch-feedback"
                    )
                    
                    export_data_btn = gr.Button(
                        "📥 Export Data",
                        variant="secondary",
                        elem_classes="touch-feedback"
                    )
    
    # Quick Examples Section - Mobile Optimized
    gr.HTML(f"""
    <div style='background: {COLORS["gradient_2"]}; 
                padding: 1rem; border-radius: 1rem; color: white; text-align: center; margin: 1rem 0;'>
        <h3 style='margin: 0 0 0.75rem 0; font-size: 1.2rem;'>🌟 Try These Foods</h3>
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin: 0.5rem 0;'>
            <div style='padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem; text-align: center;'>
                <div style='font-size: 1.5rem;'>🍕</div>
                <div style='font-size: 0.9rem;'>Pizza</div>
            </div>
            <div style='padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem; text-align: center;'>
                <div style='font-size: 1.5rem;'>🍣</div>
                <div style='font-size: 0.9rem;'>Sushi</div>
            </div>
            <div style='padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem; text-align: center;'>
                <div style='font-size: 1.5rem;'>🍔</div>
                <div style='font-size: 0.9rem;'>Burger</div>
            </div>
            <div style='padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem; text-align: center;'>
                <div style='font-size: 1.5rem;'>🍜</div>
                <div style='font-size: 0.9rem;'>Ramen</div>
            </div>
        </div>
        <p style='font-size: 0.8rem; opacity: 0.9; margin-top: 0.5rem;'>
            Don't have a photo? Search Google Images for any food!
        </p>
    </div>
    """)
    
    # Footer - Mobile Optimized
    gr.HTML(f"""
    <div style='text-align: center; padding: 1.5rem 0.5rem; background: linear-gradient(135deg, {COLORS["light"]} 0%, {COLORS["light_dark"]} 100%); 
                border-radius: 1rem; margin-top: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
        <h3 style='margin: 0 0 0.75rem 0; color: {COLORS["dark"]}; font-size: 1.2rem;'>🍕 Food Vision AI</h3>
        <p style='margin: 0.25rem 0; color: {COLORS["muted"]}; font-size: 0.9rem;'>
            <strong>Powered by:</strong> EfficientNet-B0 | PyTorch | Gradio
        </p>
        <p style='margin: 0.75rem 0 0 0; color: {COLORS["muted"]}; font-size: 0.8rem;'>
            🎯 81.10% Accuracy • ⚡ Real-time • 🌍 101 Foods
        </p>
    </div>
    """)
    
    # ==================== EVENT HANDLERS ====================
    
    # Main prediction function - Mobile Optimized
    def process_prediction(image):
        results, nutrition_html, recipe_html, analysis, shareable = predict_food(image)
        
        # Update meal history
        try:
            with sqlite3.connect('food_vision.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT food_name, calories, TIME(timestamp) as time FROM meals 
                WHERE user_id = ? AND DATE(timestamp) = DATE('now')
                ORDER BY timestamp DESC
                LIMIT 10
                ''', (1,))
                meal_data = cursor.fetchall()
        except Exception as e:
            print(f"Error updating meal history: {e}")
            meal_data = []
        
        # Update daily progress
        daily_intake = get_daily_calorie_intake()
        remaining = USER_PROFILE["daily_calorie_goal"] - daily_intake
        progress_percent = min(100, (daily_intake / USER_PROFILE["daily_calorie_goal"]) * 100)
        
        daily_progress = f"""
        ### 📊 Today's Progress
        - **Calories:** {daily_intake} / {USER_PROFILE["daily_calorie_goal"]}
        - **Remaining:** {remaining}
        - **Progress:** {progress_percent:.1f}%
        
        {'🎉 Goal reached!' if remaining <= 0 else '💪 Keep going!'}
        """
        
        # Update progress bar
        progress_bar_html = f"""
        <div class='progress-container'>
            <div class='progress-bar' style='width: {progress_percent}%; background: {COLORS["gradient_1"]};'></div>
        </div>
        <p style='text-align: center; margin-top: 0.5rem;'>{progress_percent:.1f}% Complete</p>
        """
        
        # Update recommendations
        recommendations = get_meal_recommendations()
        
        # Update weekly chart
        weekly_fig = get_weekly_nutrition_chart()
        
        return (
            results, 
            nutrition_html, 
            recipe_html, 
            analysis, 
            shareable,
            meal_data,
            daily_progress,
            progress_bar_html,
            recommendations,
            weekly_fig
        )
    
    # Connect main prediction button - Mobile Optimized
    submit_btn.click(
        fn=process_prediction,
        inputs=image_input,
        outputs=[
            predictions_output, 
            nutrition_output, 
            recipe_output, 
            analysis_output,
            shareable_output,
            meal_history_display,
            daily_progress_display,
            progress_bar_html,
            recommendations_display,
            weekly_chart
        ]
    )
    
    # Auto-predict on upload - Mobile Optimized
    image_input.change(
        fn=process_prediction,
        inputs=image_input,
        outputs=[
            predictions_output, 
            nutrition_output, 
            recipe_output, 
            analysis_output,
            shareable_output,
            meal_history_display,
            daily_progress_display,
            progress_bar_html,
            recommendations_display,
            weekly_chart
        ]
    )
    
    # Refresh meal history - Mobile Optimized
    def refresh_meal_history():
        # Get meal history from database
        try:
            with sqlite3.connect('food_vision.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT food_name, calories, TIME(timestamp) as time FROM meals 
                WHERE user_id = ? AND DATE(timestamp) = DATE('now')
                ORDER BY timestamp DESC
                LIMIT 10
                ''', (1,))
                meal_data = cursor.fetchall()
        except Exception as e:
            print(f"Error refreshing meal history: {e}")
            meal_data = []
        
        # Update daily progress
        daily_intake = get_daily_calorie_intake()
        remaining = USER_PROFILE["daily_calorie_goal"] - daily_intake
        progress_percent = min(100, (daily_intake / USER_PROFILE["daily_calorie_goal"]) * 100)
        
        daily_progress = f"""
        ### 📊 Today's Progress
        - **Calories:** {daily_intake} / {USER_PROFILE["daily_calorie_goal"]}
        - **Remaining:** {remaining}
        - **Progress:** {progress_percent:.1f}%
        
        {'🎉 Goal reached!' if remaining <= 0 else '💪 Keep going!'}
        """
        
        # Update progress bar
        progress_bar_html = f"""
        <div class='progress-container'>
            <div class='progress-bar' style='width: {progress_percent}%; background: {COLORS["gradient_1"]};'></div>
        </div>
        <p style='text-align: center; margin-top: 0.5rem;'>{progress_percent:.1f}% Complete</p>
        """
        
        # Update recommendations
        recommendations = get_meal_recommendations()
        
        # Update weekly chart
        weekly_fig = get_weekly_nutrition_chart()
        
        return meal_data, daily_progress, progress_bar_html, recommendations, weekly_fig
    
    refresh_btn.click(
        fn=refresh_meal_history,
        inputs=[],
        outputs=[
            meal_history_display,
            daily_progress_display,
            progress_bar_html,
            recommendations_display,
            weekly_chart
        ]
    )
    
    # Export nutrition report - Mobile Optimized
    def export_report():
        filename, csv = export_nutrition_report()
        if filename:
            return {"value": filename, "visible": True}
        else:
            return {"visible": False}
    
    export_btn.click(
        fn=export_report,
        inputs=[],
        outputs=[export_file]
    )
    
    # Water intake functions - Mobile Optimized
    def update_water_visual():
        water_intake = get_water_intake_today()
        water_goal = USER_PROFILE["water_goal_ml"]
        percent = min(100, (water_intake / water_goal) * 100)
        
        return f"""
        <div class='water-container'>
            <div class='water-fill' style='height: {percent}%; background: linear-gradient(180deg, {COLORS["info"]} 0%, {COLORS["primary"]} 100%);'></div>
        </div>
        <p style='text-align: center; margin-top: 0.5rem;'>{water_intake} ml / {water_goal} ml ({percent:.1f}%)</p>
        """
    
    def add_water(amount):
        add_water_intake(amount)
        return update_water_visual()
    
    water_250_btn.click(
        fn=lambda: add_water(250),
        inputs=[],
        outputs=[water_visual]
    )
    
    
    water_500_btn.click(
        fn=lambda: add_water(500),
        inputs=[],
        outputs=[water_visual]
    )
    
    water_1000_btn.click(
        fn=lambda: add_water(1000),
        inputs=[],
        outputs=[water_visual]
    )
    
    add_water_btn.click(
        fn=add_water,
        inputs=[custom_water_input],
        outputs=[water_visual]
    )
    
    def set_water_goal(goal):
        USER_PROFILE["water_goal_ml"] = goal
        return update_water_visual()
    
    set_water_goal_btn.click(
        fn=set_water_goal,
        inputs=[water_goal_input],
        outputs=[water_visual]
    )
    
    # Exercise tracking - Mobile Optimized
    def update_exercise_list():
        exercise_data = get_exercise_today()
        return exercise_data
    
    def add_exercise_to_list(activity, duration, calories):
        add_exercise(activity, duration, calories)
        return update_exercise_list()
    
    add_exercise_btn.click(
        fn=add_exercise_to_list,
        inputs=[exercise_type, exercise_duration, exercise_calories],
        outputs=[exercise_list]
    )
    
    # Recipe search - Mobile Optimized
    def search_recipes(query):
        if not query:
            return f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Enter a food name</div>"
        
        # Find matching recipes
        matches = []
        for key, recipe in RECIPE_DB.items():
            if query.lower() in key.lower() or query.lower() in recipe["name"].lower():
                matches.append((key, recipe))
        
        if not matches:
            return f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>No recipes for '{query}'</div>"
        
        # Display the first match
        key, recipe = matches[0]
        food_name = key.replace('_', ' ').title()
        
        recipe_html = f"""
        <div style='background: {COLORS["gradient_2"]}; 
                    padding: 1.5rem; border-radius: 1rem; color: white; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            
            <h2 style='margin: 0 0 1rem 0; text-align: center; font-size: 1.5rem;'>
                🍳 {recipe['name']}
            </h2>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; text-align: center;'>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>⏱️</div>
                    <div>{recipe['time']}</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>📊</div>
                    <div>{recipe['difficulty']}</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>🍽️</div>
                    <div>{recipe['servings']} servings</div>
                </div>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                <h3 style='margin: 0 0 0.5rem 0;'>🥘 Ingredients</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; font-size: 0.8rem;'>
                """
        
        for ingredient in recipe['ingredients']:
            recipe_html += f"<div>• {ingredient}</div>"
        
        recipe_html += f"""
                </div>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='margin: 0 0 0.5rem 0;'>📝 Instructions</h3>
                <ol style='padding-left: 1rem; font-size: 0.8rem;'>
                """
        
        for step in recipe['steps']:
            recipe_html += f"<li>{step}</li>"
        
        recipe_html += f"""
                </ol>
            </div>
            
            <div style='text-align: center; margin-top: 1rem; font-size: 0.8rem; opacity: 0.9;'>
                🔥 {recipe['calories']} calories per serving
            </div>
        </div>
        """
        
        return recipe_html
    
    search_btn.click(
        fn=search_recipes,
        inputs=recipe_search,
        outputs=recipe_display
    )
    
    # Category buttons - Mobile Optimized
    def get_recipes_by_category(category):
        # Define categories
        categories = {
            "Breakfast": ["pancakes", "french_toast", "omelette"],
            "Lunch": ["salad", "burger", "tacos"],
            "Dinner": ["pasta", "pizza", "chicken_curry"],
            "Dessert": ["ice_cream", "apple_pie"]
        }
        
        if category not in categories:
            return f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Category not found</div>"
        
        # Get a random recipe from the category
        food_key = random.choice(categories[category])
        recipe = RECIPE_DB.get(food_key, {})
        
        if not recipe:
            return f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>No recipes in {category}</div>"
        
        food_name = food_key.replace('_', ' ').title()
        
        recipe_html = f"""
        <div style='background: {COLORS["gradient_2"]}; 
                    padding: 1.5rem; border-radius: 1rem; color: white; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            
            <h2 style='margin: 0 0 1rem 0; text-align: center; font-size: 1.5rem;'>
                🍳 {recipe['name']}
            </h2>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem; text-align: center;'>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>⏱️</div>
                    <div>{recipe['time']}</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>📊</div>
                    <div>{recipe['difficulty']}</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 0.5rem;'>
                    <div style='font-size: 1.2rem; font-weight: bold;'>🍽️</div>
                    <div>{recipe['servings']} servings</div>
                </div>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                <h3 style='margin: 0 0 0.5rem 0;'>🥘 Ingredients</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; font-size: 0.8rem;'>
                """
        
        for ingredient in recipe['ingredients']:
            recipe_html += f"<div>• {ingredient}</div>"
        
        recipe_html += f"""
                </div>
            </div>
            
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 0.5rem;'>
                <h3 style='margin: 0 0 0.5rem 0;'>📝 Instructions</h3>
                <ol style='padding-left: 1rem; font-size: 0.8rem;'>
                """
        
        for step in recipe['steps']:
            recipe_html += f"<li>{step}</li>"
        
        recipe_html += f"""
                </ol>
            </div>
            
            <div style='text-align: center; margin-top: 1rem; font-size: 0.8rem; opacity: 0.9;'>
                🔥 {recipe['calories']} calories per serving
            </div>
        </div>
        """
        
        return recipe_html
    
    breakfast_btn.click(
        fn=lambda: get_recipes_by_category("Breakfast"),
        inputs=[],
        outputs=recipe_display
    )
    
    lunch_btn.click(
        fn=lambda: get_recipes_by_category("Lunch"),
        inputs=[],
        outputs=recipe_display
    )
    
    dinner_btn.click(
        fn=lambda: get_recipes_by_category("Dinner"),
        inputs=[],
        outputs=recipe_display
    )
    
    dessert_btn.click(
        fn=lambda: get_recipes_by_category("Dessert"),
        inputs=[],
        outputs=recipe_display
    )
    
    # Food comparison - Mobile Optimized
    def compare_selected_foods(food1, food2):
        fig, text = compare_foods(food1, food2)
        return fig, text
    
    compare_btn.click(
        fn=compare_selected_foods,
        inputs=[food1_dropdown, food2_dropdown],
        outputs=[comparison_chart, comparison_text]
    )
    
    # Meal planner - Mobile Optimized
    def generate_meal_plan_handler(meal_type, calorie_target):
        return generate_meal_plan(meal_type, calorie_target)
    
    generate_plan_btn.click(
        fn=generate_meal_plan_handler,
        inputs=[meal_type, calorie_target],
        outputs=[meal_plan_output]
    )
    
    # Shopping list - Mobile Optimized
    def update_shopping_list_display():
        items = get_shopping_list()
        
        if not items:
            return f"<div style='text-align: center; padding: 2rem; color: {COLORS['muted']};'>Your list is empty</div>"
        
        html = "<div class='shopping-list'>"
        
        for item_id, item_name, quantity, checked in items:
            checked_class = "checked" if checked else ""
            checked_attr = "checked" if checked else ""
            
            html += f"""
            <div class='shopping-item {checked_class}'>
                <input type='checkbox' class='shopping-checkbox' {checked_attr} onchange='toggleItem({item_id})'>
                <span>{item_name} - {quantity}</span>
                <button onclick='deleteItem({item_id})' style='margin-left: auto; background: {COLORS["danger"]}; color: white; border: none; border-radius: 0.25rem; padding: 0.25rem 0.5rem; font-size: 0.8rem;'>Delete</button>
            </div>
            """
        
        html += "</div>"
        
        return html
    
    def add_shopping_item(item_name, quantity):
        add_to_shopping_list(item_name, quantity)
        return update_shopping_list_display()
    
    def clear_shopping_list():
        try:
            with sqlite3.connect('food_vision.db') as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM shopping_list WHERE user_id = ?', (1,))
                conn.commit()
            return update_shopping_list_display()
        except Exception as e:
            print(f"Error clearing shopping list: {e}")
            return update_shopping_list_display()
    
    add_item_btn.click(
        fn=add_shopping_item,
        inputs=[item_name, item_quantity],
        outputs=[shopping_list_display]
    )
    
    refresh_list_btn.click(
        fn=update_shopping_list_display,
        inputs=[],
        outputs=[shopping_list_display]
    )
    
    clear_list_btn.click(
        fn=clear_shopping_list,
        inputs=[],
        outputs=[shopping_list_display]
    )
    
    # BMI calculator - Mobile Optimized
    def calculate_bmi_result(height, weight):
        bmi, category, color = calculate_bmi(height, weight)
        
        # Determine CSS class
        if category == "Underweight":
            css_class = "bmi-underweight"
        elif category == "Normal weight":
            css_class = "bmi-normal"
        elif category == "Overweight":
            css_class = "bmi-overweight"
        else:
            css_class = "bmi-obese"
        
        return f"""
        <div class='bmi-result {css_class}'>
            <h3>Your BMI: {bmi}</h3>
            <p>Category: {category}</p>
        </div>
        """
    
    calculate_bmi_btn.click(
        fn=calculate_bmi_result,
        inputs=[height_input, weight_input],
        outputs=[bmi_result]
    )
    
    # Calorie calculator - Mobile Optimized
    def calculate_daily_calories_result(age, gender, activity_level):
        # Simplify activity level for calculation
        if "Sedentary" in activity_level:
            level = "sedentary"
        elif "Light" in activity_level:
            level = "light"
        elif "Moderate" in activity_level:
            level = "moderate"
        elif "Active" in activity_level and "Very" not in activity_level:
            level = "active"
        else:
            level = "very_active"
        
        calories = calculate_daily_calories(
            USER_PROFILE["height_cm"],
            USER_PROFILE["weight_kg"],
            age,
            gender.lower(),
            level
        )
        
        return f"""
        <div style='background: {COLORS["gradient_3"]}; 
                    padding: 1.5rem; border-radius: 1rem; color: white; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2); text-align: center;'>
            <h3>Your Daily Calories</h3>
            <p style='font-size: 1.5rem; font-weight: bold;'>{calories}</p>
            <p>Based on your profile</p>
        </div>
        """
    
    calculate_calories_btn.click(
        fn=calculate_daily_calories_result,
        inputs=[age_input, gender_input, activity_input],
        outputs=[calorie_result]
    )
    
    # Save user profile - Mobile Optimized
    def save_profile(name, email, calorie_goal, dietary_prefs, allergies):
        USER_PROFILE["name"] = name
        USER_PROFILE["email"] = email
        USER_PROFILE["daily_calorie_goal"] = calorie_goal
        USER_PROFILE["dietary_preferences"] = dietary_prefs
        USER_PROFILE["allergies"] = allergies
        
        return {"value": "✅ Profile saved!", "visible": True}
    
    save_profile_btn.click(
        fn=save_profile,
        inputs=[name_input, email_input, calorie_goal_input, dietary_preferences, allergies],
        outputs=[status_output]
    )
    
    # Save app settings - Mobile Optimized
    def save_settings(theme, notifications, auto_save):
        # In a real app, these would be saved to a database or file
        return {"value": "✅ Settings saved!", "visible": True}
    
    save_settings_btn.click(
        fn=save_settings,
        inputs=[theme_selector, notifications, auto_save],
        outputs=[status_output]
    )
    
    # Clear all data - Mobile Optimized
    def clear_all_data():
        global MEAL_HISTORY
        MEAL_HISTORY = []
        
        # Clear database
        try:
            with sqlite3.connect('food_vision.db') as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM meals WHERE user_id = ?', (1,))
                cursor.execute('DELETE FROM water_intake WHERE user_id = ?', (1,))
                cursor.execute('DELETE FROM exercise WHERE user_id = ?', (1,))
                cursor.execute('DELETE FROM shopping_list WHERE user_id = ?', (1,))
                conn.commit()
            
            return {"value": "✅ All data cleared!", "visible": True}
        except Exception as e:
            print(f"Error clearing data: {e}")
            return {"value": f"Error clearing data: {str(e)}", "visible": True}
    
    clear_data_btn.click(
        fn=clear_all_data,
        inputs=[],
        outputs=[status_output]
    )

    # Export data button in settings section
    export_data_btn.click(
        fn=export_report,
        inputs=[],
        outputs=[export_file]
    )

# ==================== LAUNCH ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 LAUNCHING FOOD VISION AI - ULTIMATE PRO VERSION")
    print("="*60)
    print(f"✓ Model loaded: EfficientNet-B0")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Classes: {NUM_CLASSES}")
    print(f"✓ Accuracy: 81.10% (Top-1), 95.64% (Top-5)")
    print(f"✓ Nutrition database: {len(NUTRITION_DB)} foods")
    print(f"✓ Recipe database: {len(RECIPE_DB)} recipes")
    print(f"✓ Database initialized: food_vision.db")
    print("="*60)
    print("\n🎉 App ready! Opening in browser...\n")
    
    demo.launch(
        share=True,  # Create public URL
        inbrowser=True,  # Auto-open browser
        show_error=True,
        server_name="0.0.0.0"
    )