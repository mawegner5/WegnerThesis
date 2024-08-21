import pandas as pd
import os

# Define the threshold hyperparameter
threshold = 1.0  # Example threshold value; adjust this as needed

# Load the dataset
input_csv_path = '/root/.ipython/WegnerThesis/data/generated_data/class_attributes.csv'
df = pd.read_csv(input_csv_path)

# Rename the 'Unnamed: 0' column to 'bird_species'
df.rename(columns={'Unnamed: 0': 'bird_species'}, inplace=True)

# Create a dictionary to map attribute columns to readable descriptions
attribute_to_description = {
    # Bin: Bill Shape
    'has_bill_shape::curved_(up_or_down)': 'curved (up or down)',
    'has_bill_shape::dagger': 'dagger-shaped',
    'has_bill_shape::hooked': 'hooked',
    'has_bill_shape::needle': 'needle-shaped',
    'has_bill_shape::hooked_seabird': 'hooked seabird',
    'has_bill_shape::spatulate': 'spatulated',
    'has_bill_shape::all-purpose': 'all-purpose',
    'has_bill_shape::cone': 'cone-shaped',
    'has_bill_shape::specialized': 'specialized',
    
    # Bin: Wing Color
    'has_wing_color::blue': 'blue',
    'has_wing_color::brown': 'brown',
    'has_wing_color::iridescent': 'iridescent',
    'has_wing_color::purple': 'purple',
    'has_wing_color::rufous': 'rufous',
    'has_wing_color::grey': 'grey',
    'has_wing_color::yellow': 'yellow',
    'has_wing_color::olive': 'olive',
    'has_wing_color::green': 'green',
    'has_wing_color::pink': 'pink',
    'has_wing_color::orange': 'orange',
    'has_wing_color::black': 'black',
    'has_wing_color::white': 'white',
    'has_wing_color::red': 'red',
    'has_wing_color::buff': 'buff',

    # Bin: Upperparts Color
    'has_upperparts_color::blue': 'blue',
    'has_upperparts_color::brown': 'brown',
    'has_upperparts_color::iridescent': 'iridescent',
    'has_upperparts_color::purple': 'purple',
    'has_upperparts_color::rufous': 'rufous',
    'has_upperparts_color::grey': 'grey',
    'has_upperparts_color::yellow': 'yellow',
    'has_upperparts_color::olive': 'olive',
    'has_upperparts_color::green': 'green',
    'has_upperparts_color::pink': 'pink',
    'has_upperparts_color::orange': 'orange',
    'has_upperparts_color::black': 'black',
    'has_upperparts_color::white': 'white',
    'has_upperparts_color::red': 'red',
    'has_upperparts_color::buff': 'buff',

    # Bin: Underparts Color
    'has_underparts_color::blue': 'blue',
    'has_underparts_color::brown': 'brown',
    'has_underparts_color::iridescent': 'iridescent',
    'has_underparts_color::purple': 'purple',
    'has_underparts_color::rufous': 'rufous',
    'has_underparts_color::grey': 'grey',
    'has_underparts_color::yellow': 'yellow',
    'has_underparts_color::olive': 'olive',
    'has_underparts_color::green': 'green',
    'has_underparts_color::pink': 'pink',
    'has_underparts_color::orange': 'orange',
    'has_underparts_color::black': 'black',
    'has_underparts_color::white': 'white',
    'has_underparts_color::red': 'red',
    'has_underparts_color::buff': 'buff',

    # Bin: Back Color
    'has_back_color::blue': 'blue',
    'has_back_color::brown': 'brown',
    'has_back_color::iridescent': 'iridescent',
    'has_back_color::purple': 'purple',
    'has_back_color::rufous': 'rufous',
    'has_back_color::grey': 'grey',
    'has_back_color::yellow': 'yellow',
    'has_back_color::olive': 'olive',
    'has_back_color::green': 'green',
    'has_back_color::pink': 'pink',
    'has_back_color::orange': 'orange',
    'has_back_color::black': 'black',
    'has_back_color::white': 'white',
    'has_back_color::red': 'red',
    'has_back_color::buff': 'buff',

    # Bin: Tail Shape
    'has_tail_shape::forked_tail': 'forked',
    'has_tail_shape::rounded_tail': 'rounded',
    'has_tail_shape::notched_tail': 'notched',
    'has_tail_shape::fan-shaped_tail': 'fan-shaped',
    'has_tail_shape::pointed_tail': 'pointed',
    'has_tail_shape::squared_tail': 'squared',

    # Bin: Upper Tail Color
    'has_upper_tail_color::blue': 'blue',
    'has_upper_tail_color::brown': 'brown',
    'has_upper_tail_color::iridescent': 'iridescent',
    'has_upper_tail_color::purple': 'purple',
    'has_upper_tail_color::rufous': 'rufous',
    'has_upper_tail_color::grey': 'grey',
    'has_upper_tail_color::yellow': 'yellow',
    'has_upper_tail_color::olive': 'olive',
    'has_upper_tail_color::green': 'green',
    'has_upper_tail_color::pink': 'pink',
    'has_upper_tail_color::orange': 'orange',
    'has_upper_tail_color::black': 'black',
    'has_upper_tail_color::white': 'white',
    'has_upper_tail_color::red': 'red',
    'has_upper_tail_color::buff': 'buff',

    # Bin: Head Pattern
    'has_head_pattern::spotted': 'spotted',
    'has_head_pattern::malar': 'malar',
    'has_head_pattern::crested': 'crested',
    'has_head_pattern::masked': 'masked',
    'has_head_pattern::unique_pattern': 'unique',
    'has_head_pattern::eyebrow': 'eyebrow',
    'has_head_pattern::eyering': 'eyering',
    'has_head_pattern::plain': 'plain',
    'has_head_pattern::eyeline': 'eyeline',
    'has_head_pattern::striped': 'striped',
    'has_head_pattern::capped': 'capped',

    # Bin: Breast Color
    'has_breast_color::blue': 'blue',
    'has_breast_color::brown': 'brown',
    'has_breast_color::iridescent': 'iridescent',
    'has_breast_color::purple': 'purple',
    'has_breast_color::rufous': 'rufous',
    'has_breast_color::grey': 'grey',
    'has_breast_color::yellow': 'yellow',
    'has_breast_color::olive': 'olive',
    'has_breast_color::green': 'green',
    'has_breast_color::pink': 'pink',
    'has_breast_color::orange': 'orange',
    'has_breast_color::black': 'black',
    'has_breast_color::white': 'white',
    'has_breast_color::red': 'red',
    'has_breast_color::buff': 'buff',

    # Bin: Throat Color
    'has_throat_color::blue': 'blue',
    'has_throat_color::brown': 'brown',
    'has_throat_color::iridescent': 'iridescent',
    'has_throat_color::purple': 'purple',
    'has_throat_color::rufous': 'rufous',
    'has_throat_color::grey': 'grey',
    'has_throat_color::yellow': 'yellow',
    'has_throat_color::olive': 'olive',
    'has_throat_color::green': 'green',
    'has_throat_color::pink': 'pink',
    'has_throat_color::orange': 'orange',
    'has_throat_color::black': 'black',
    'has_throat_color::white': 'white',
    'has_throat_color::red': 'red',
    'has_throat_color::buff': 'buff',

    # Bin: Eye Color
    'has_eye_color::blue': 'blue',
    'has_eye_color::brown': 'brown',
    'has_eye_color::purple': 'purple',
    'has_eye_color::rufous': 'rufous',
    'has_eye_color::grey': 'grey',
    'has_eye_color::yellow': 'yellow',
    'has_eye_color::olive': 'olive',
    'has_eye_color::green': 'green',
    'has_eye_color::pink': 'pink',
    'has_eye_color::orange': 'orange',
    'has_eye_color::black': 'black',
    'has_eye_color::white': 'white',
    'has_eye_color::red': 'red',
    'has_eye_color::buff': 'buff',

    # Bin: Bill Length
    'has_bill_length::about_the_same_as_head': 'about the same length as the head',
    'has_bill_length::longer_than_head': 'longer than the head',
    'has_bill_length::shorter_than_head': 'shorter than the head',

    # Bin: Forehead Color
    'has_forehead_color::blue': 'blue',
    'has_forehead_color::brown': 'brown',
    'has_forehead_color::iridescent': 'iridescent',
    'has_forehead_color::purple': 'purple',
    'has_forehead_color::rufous': 'rufous',
    'has_forehead_color::grey': 'grey',
    'has_forehead_color::yellow': 'yellow',
    'has_forehead_color::olive': 'olive',
    'has_forehead_color::green': 'green',
    'has_forehead_color::pink': 'pink',
    'has_forehead_color::orange': 'orange',
    'has_forehead_color::black': 'black',
    'has_forehead_color::white': 'white',
    'has_forehead_color::red': 'red',
    'has_forehead_color::buff': 'buff',

    # Bin: Under Tail Color
    'has_under_tail_color::blue': 'blue',
    'has_under_tail_color::brown': 'brown',
    'has_under_tail_color::iridescent': 'iridescent',
    'has_under_tail_color::purple': 'purple',
    'has_under_tail_color::rufous': 'rufous',
    'has_under_tail_color::grey': 'grey',
    'has_under_tail_color::yellow': 'yellow',
    'has_under_tail_color::olive': 'olive',
    'has_under_tail_color::green': 'green',
    'has_under_tail_color::pink': 'pink',
    'has_under_tail_color::orange': 'orange',
    'has_under_tail_color::black': 'black',
    'has_under_tail_color::white': 'white',
    'has_under_tail_color::red': 'red',
    'has_under_tail_color::buff': 'buff',

    # Bin: Nape Color
    'has_nape_color::blue': 'blue',
    'has_nape_color::brown': 'brown',
    'has_nape_color::iridescent': 'iridescent',
    'has_nape_color::purple': 'purple',
    'has_nape_color::rufous': 'rufous',
    'has_nape_color::grey': 'grey',
    'has_nape_color::yellow': 'yellow',
    'has_nape_color::olive': 'olive',
    'has_nape_color::green': 'green',
    'has_nape_color::pink': 'pink',
    'has_nape_color::orange': 'orange',
    'has_nape_color::black': 'black',
    'has_nape_color::white': 'white',
    'has_nape_color::red': 'red',
    'has_nape_color::buff': 'buff',

    # Bin: Belly Color
    'has_belly_color::blue': 'blue',
    'has_belly_color::brown': 'brown',
    'has_belly_color::iridescent': 'iridescent',
    'has_belly_color::purple': 'purple',
    'has_belly_color::rufous': 'rufous',
    'has_belly_color::grey': 'grey',
    'has_belly_color::yellow': 'yellow',
    'has_belly_color::olive': 'olive',
    'has_belly_color::green': 'green',
    'has_belly_color::pink': 'pink',
    'has_belly_color::orange': 'orange',
    'has_belly_color::black': 'black',
    'has_belly_color::white': 'white',
    'has_belly_color::red': 'red',
    'has_belly_color::buff': 'buff',

    # Bin: Wing Shape
    'has_wing_shape::rounded-wings': 'rounded wings',
    'has_wing_shape::pointed-wings': 'pointed wings',
    'has_wing_shape::broad-wings': 'broad wings',
    'has_wing_shape::tapered-wings': 'tapered wings',
    'has_wing_shape::long-wings': 'long wings',

    # Bin: Size
    'has_size::large_(16_-_32_in)': 'large size (16 - 32 inches)',
    'has_size::small_(5_-_9_in)': 'small size (5 - 9 inches)',
    'has_size::very_large_(32_-_72_in)': 'very large size (32 - 72 inches)',
    'has_size::medium_(9_-_16_in)': 'medium size (9 - 16 inches)',
    'has_size::very_small_(3_-_5_in)': 'very small size (3 - 5 inches)',

    # Bin: Shape
    'has_shape::upright-perching_water-like': 'upright-perching water-like shape',
    'has_shape::chicken-like-marsh': 'chicken-like marsh shape',
    'has_shape::long-legged-like': 'long-legged shape',
    'has_shape::duck-like': 'duck-like shape',
    'has_shape::owl-like': 'owl-like shape',
    'has_shape::gull-like': 'gull-like shape',
    'has_shape::hummingbird-like': 'hummingbird-like shape',
    'has_shape::pigeon-like': 'pigeon-like shape',
    'has_shape::tree-clinging-like': 'tree-clinging shape',
    'has_shape::hawk-like': 'hawk-like shape',
    'has_shape::sandpiper-like': 'sandpiper-like shape',
    'has_shape::upland-ground-like': 'upland-ground shape',
    'has_shape::swallow-like': 'swallow-like shape',
    'has_shape::perching-like': 'perching-like shape',

    # Bin: Back Pattern
    'has_back_pattern::solid': 'solid back pattern',
    'has_back_pattern::spotted': 'spotted back pattern',
    'has_back_pattern::striped': 'striped back pattern',
    'has_back_pattern::multi-colored': 'multi-colored back pattern',

    # Bin: Tail Pattern
    'has_tail_pattern::solid': 'solid tail pattern',
    'has_tail_pattern::spotted': 'spotted tail pattern',
    'has_tail_pattern::striped': 'striped tail pattern',
    'has_tail_pattern::multi-colored': 'multi-colored tail pattern',

    # Bin: Belly Pattern
    'has_belly_pattern::solid': 'solid belly pattern',
    'has_belly_pattern::spotted': 'spotted belly pattern',
    'has_belly_pattern::striped': 'striped belly pattern',
    'has_belly_pattern::multi-colored': 'multi-colored belly pattern',

    # Bin: Primary Color
    'has_primary_color::blue': 'primary color of blue',
    'has_primary_color::brown': 'primary color of brown',
    'has_primary_color::iridescent': 'primary color of iridescent',
    'has_primary_color::purple': 'primary color of purple',
    'has_primary_color::rufous': 'primary color of rufous',
    'has_primary_color::grey': 'primary color of grey',
    'has_primary_color::yellow': 'primary color of yellow',
    'has_primary_color::olive': 'primary color of olive',
    'has_primary_color::green': 'primary color of green',
    'has_primary_color::pink': 'primary color of pink',
    'has_primary_color::orange': 'primary color of orange',
    'has_primary_color::black': 'primary color of black',
    'has_primary_color::white': 'primary color of white',
    'has_primary_color::red': 'primary color of red',
    'has_primary_color::buff': 'primary color of buff',

    # Bin: Leg Color
    'has_leg_color::blue': 'blue legs',
    'has_leg_color::brown': 'brown legs',
    'has_leg_color::iridescent': 'iridescent legs',
    'has_leg_color::purple': 'purple legs',
    'has_leg_color::rufous': 'rufous legs',
    'has_leg_color::grey': 'grey legs',
    'has_leg_color::yellow': 'yellow legs',
    'has_leg_color::olive': 'olive legs',
    'has_leg_color::green': 'green legs',
    'has_leg_color::pink': 'pink legs',
    'has_leg_color::orange': 'orange legs',
    'has_leg_color::black': 'black legs',
    'has_leg_color::white': 'white legs',
    'has_leg_color::red': 'red legs',
    'has_leg_color::buff': 'buff legs',

    # Bin: Bill Color
    'has_bill_color::blue': 'a blue bill',
    'has_bill_color::brown': 'a brown bill',
    'has_bill_color::iridescent': 'an iridescent bill',
    'has_bill_color::purple': 'a purple bill',
    'has_bill_color::rufous': 'a rufous bill',
    'has_bill_color::grey': 'a grey bill',
    'has_bill_color::yellow': 'a yellow bill',
    'has_bill_color::olive': 'an olive bill',
    'has_bill_color::green': 'a green bill',
    'has_bill_color::pink': 'a pink bill',
    'has_bill_color::orange': 'an orange bill',
    'has_bill_color::black': 'a black bill',
    'has_bill_color::white': 'a white bill',
    'has_bill_color::red': 'a red bill',
    'has_bill_color::buff': 'a buff bill',

    # Bin: Crown Color
    'has_crown_color::blue': 'a blue crown',
    'has_crown_color::brown': 'a brown crown',
    'has_crown_color::iridescent': 'an iridescent crown',
    'has_crown_color::purple': 'a purple crown',
    'has_crown_color::rufous': 'a rufous crown',
    'has_crown_color::grey': 'a grey crown',
    'has_crown_color::yellow': 'a yellow crown',
    'has_crown_color::olive': 'an olive crown',
    'has_crown_color::green': 'a green crown',
    'has_crown_color::pink': 'a pink crown',
    'has_crown_color::orange': 'an orange crown',
    'has_crown_color::black': 'a black crown',
    'has_crown_color::white': 'a white crown',
    'has_crown_color::red': 'a red crown',
    'has_crown_color::buff': 'a buff crown',

    # Bin: Wing Pattern
    'has_wing_pattern::solid': 'solid wing pattern',
    'has_wing_pattern::spotted': 'spotted wing pattern',
    'has_wing_pattern::striped': 'striped wing pattern',
    'has_wing_pattern::multi-colored': 'multi-colored wing pattern',
}


# Function to bin and summarize attributes into readable descriptions
def bin_attributes(row, threshold):
    # Initialize bins
    bill_descriptions = []
    wing_colors = []
    # Add other bins as needed (e.g., upperparts_colors, back_colors)
    
    for attribute, score in row.items():
        if attribute in attribute_to_description and score >= threshold:
            description = attribute_to_description[attribute]
            
            # Append descriptions to appropriate bins
            if 'bill_shape' in attribute:
                bill_descriptions.append(description)
            elif 'wing_color' in attribute:
                wing_colors.append(description)
            # Add conditions for other bins
    
    # Combine the descriptions
    descriptions = []
    if bill_descriptions:
        descriptions.append(f"its beak is {' and '.join(bill_descriptions)}")
    if wing_colors:
        descriptions.append(f"its wings are {', '.join(wing_colors)}")
    # Add combined descriptions for other bins
    
    return '. '.join(descriptions)

# Apply the conversion to each row
df['sentences'] = df.apply(bin_attributes, axis=1, threshold=threshold)

# Save the new CSV file
output_csv_path = f'/root/.ipython/WegnerThesis/data/generated_data/binned_classes_array_{threshold}.csv'
df[['bird_species', 'sentences']].to_csv(output_csv_path, index=False)
