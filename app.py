from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your pre-trained MobileNet model for food category recognition
category_model = load_model('models/mobilenet_food11.h5')

# Dictionary to map food categories to variety-specific models
variety_models = {
    'apple_pie': load_model('models/fine_grained_apple_pie_classifier.h5'),
    'cheesecake': load_model('models/fine_grained_cheesecake_classifier.h5'),
    'chicken_curry': load_model('models/fine_grained_chicken_curry_classifier.h5'),
    'french_fries': load_model('models/fine_grained_french_fries_classifier.h5'),
    'fried_rice': load_model('models/fine_grained_fried_rice_classifier.h5'),
    'hamburger': load_model('models/fine_grained_hamburger_classifier.h5'),
    'hot_dog': load_model('models/fine_grained_hot_dog_classifier.h5'),
    'ice_cream': load_model('models/fine_grained_ice_cream_classifier.h5'),
    'omelette': load_model('models/fine_grained_omelette_classifier.h5'),
    'pizza': load_model('models/fine_grained_pizza_classifier.h5'),
    'sushi': load_model('models/fine_grained_sushi_classifier.h5')
}

# Load the JSON file with recipes, ingredients, and nutrient details
with open('food_varieties.json', 'r') as f:
    food_data = json.load(f)

# Class labels for food categories (replace with your actual category labels)
class_labels = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice', 
                'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']

# Class labels for food varieties (example for apple_pie, replace with actual variety labels)
variety_labels = {
    'apple_pie': ['classic_apple_pie', 'lattice_apple_pie', 'french_apple_pie'],
    'cheesecake': ['fruit_topped_cheesecake', 'blueberry_cheesecake','mixed_berry_fruit_swirl_cheesecake',
                   'chocolate_drizzled_cheesecake','spiced_cheesecake','classic_cheesecake','berry_swirl_cheesecake',
                   'layered_cheesecake','decorative_cheesecake','plain_cheesecake'],
    'chicken_curry':['butter_chicken_curry','spicy_tandoori_hicken_curry','south_indian_chicken_curry','kadhai_chicken_curry',
                     'chicken_tikka_masala','spicy_chicken_vindaloo'],
    'french_fries':['topped_fries','fast_food_french_fries','classic_french_fries','crinkle_cut_fries','shoestring_fries',
                    'sweet_potato_fries'],
    'hamburger':['classic_cheeseburger','bacon_cheeseburger','double_patty_burger','gourmet_truffle_burger','barbecue_bacon_burger',
                 'veggie_burger','spicy_jalapeño_burger','fully_stacked_burger'],
    
    'fried_rice':['vegetable_fried_rice','chicken_fried_rice','egg_fried_rice','seafood_fried_rice','beef_fried_rice','shrimp_fried_rice',
                  'pork_fried_rice','spicy_fried_rice'],
    
    'hot_dog':['chili_hot_dogs','chicago_style_hot_dogs','classic_american_hot_dogs','american_mustard_and_ketchup_hot_dogs',
               'bratwurst_hot_dogs','new_york_style_hot_dogs','bacon_wrapped_hot_dogs'],
    'pizza':['classic_margherita_pizza','carnivore_feast_pizza','gourmet_artisan_pizza'],
    'sushi':['maki_sushi','nigiri_sushi','sashimi_sushi','temaki_sushi','futomaki_sushi','uramaki_sushi','chirashi_sushi']
    
}

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/recognize', methods=['POST'])
@app.route("/predict", methods=["GET", "POST"])
def recognize_food():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Step 1: Recognize the food category
        img = process_image(filepath)
        category_predictions = category_model.predict(img)
        predicted_category = np.argmax(category_predictions, axis=1)[0]
        predicted_probability = np.max(category_predictions)
        food_name = class_labels[predicted_category]
    
        # Step 2: Recognize the variety within the food category
        variety_model = variety_models[food_name]
        variety_predictions = variety_model.predict(img)
        predicted_variety = np.argmax(variety_predictions, axis=1)[0]
        variety_name = variety_labels[food_name][predicted_variety]

        # Step 3: Fetch recipe, ingredients, and nutrient details from the JSON
        food_variety_details = food_data[food_name][variety_name]

        return jsonify({
            'foodName': food_name,
            'foodVariety': variety_name,
            'probability': str(predicted_probability),
            'ingredients': food_variety_details['ingredients'],
            'recipe': food_variety_details['recipe'],
            'nutrients': food_variety_details['nutrients']
        })



if __name__ == '__main__':
    app.run(debug=True)
