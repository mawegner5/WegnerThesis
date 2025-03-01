# Create a dictionary to map attribute columns to readable descriptions
attribute_to_description = {
    'has_bill_shape::curved_(up_or_down)': 'a curved bill (up or down)',
    'has_bill_shape::dagger': 'a dagger-shaped bill',
    'has_bill_shape::hooked': 'a hooked bill',
    'has_bill_shape::needle': 'a needle-shaped bill',
    'has_bill_shape::hooked_seabird': 'a hooked seabird bill',
    'has_bill_shape::spatulate': 'a spatulate bill',
    'has_bill_shape::all-purpose': 'an all-purpose bill',
    'has_bill_shape::cone': 'a cone-shaped bill',
    'has_bill_shape::specialized': 'a specialized bill',
    'has_wing_color::blue': 'blue wings',
    'has_wing_color::brown': 'brown wings',
    'has_wing_color::iridescent': 'iridescent wings',
    'has_wing_color::purple': 'purple wings',
    'has_wing_color::rufous': 'rufous wings',
    'has_wing_color::grey': 'grey wings',
    'has_wing_color::yellow': 'yellow wings',
    'has_wing_color::olive': 'olive wings',
    'has_wing_color::green': 'green wings',
    'has_wing_color::pink': 'pink wings',
    'has_wing_color::orange': 'orange wings',
    'has_wing_color::black': 'black wings',
    'has_wing_color::white': 'white wings',
    'has_wing_color::red': 'red wings',
    'has_wing_color::buff': 'buff wings',
    'has_upperparts_color::blue': 'blue upperparts',
    'has_upperparts_color::brown': 'brown upperparts',
    'has_upperparts_color::iridescent': 'iridescent upperparts',
    'has_upperparts_color::purple': 'purple upperparts',
    'has_upperparts_color::rufous': 'rufous upperparts',
    'has_upperparts_color::grey': 'grey upperparts',
    'has_upperparts_color::yellow': 'yellow upperparts',
    'has_upperparts_color::olive': 'olive upperparts',
    'has_upperparts_color::green': 'green upperparts',
    'has_upperparts_color::pink': 'pink upperparts',
    'has_upperparts_color::orange': 'orange upperparts',
    'has_upperparts_color::black': 'black upperparts',
    'has_upperparts_color::white': 'white upperparts',
    'has_upperparts_color::red': 'red upperparts',
    'has_upperparts_color::buff': 'buff upperparts',
    'has_underparts_color::blue': 'blue underparts',
    'has_underparts_color::brown': 'brown underparts',
    'has_underparts_color::iridescent': 'iridescent underparts',
    'has_underparts_color::purple': 'purple underparts',
    'has_underparts_color::rufous': 'rufous underparts',
    'has_underparts_color::grey': 'grey underparts',
    'has_underparts_color::yellow': 'yellow underparts',
    'has_underparts_color::olive': 'olive underparts',
    'has_underparts_color::green': 'green underparts',
    'has_underparts_color::pink': 'pink underparts',
    'has_underparts_color::orange': 'orange underparts',
    'has_underparts_color::black': 'black underparts',
    'has_underparts_color::white': 'white underparts',
    'has_underparts_color::red': 'red underparts',
    'has_underparts_color::buff': 'buff underparts',
    'has_breast_pattern::solid': 'a solid breast pattern',
    'has_breast_pattern::spotted': 'a spotted breast pattern',
    'has_breast_pattern::striped': 'a striped breast pattern',
    'has_breast_pattern::multi-colored': 'a multi-colored breast pattern',
    'has_back_color::blue': 'a blue back',
    'has_back_color::brown': 'a brown back',
    'has_back_color::iridescent': 'an iridescent back',
    'has_back_color::purple': 'a purple back',
    'has_back_color::rufous': 'a rufous back',
    'has_back_color::grey': 'a grey back',
    'has_back_color::yellow': 'a yellow back',
    'has_back_color::olive': 'an olive back',
    'has_back_color::green': 'a green back',
    'has_back_color::pink': 'a pink back',
    'has_back_color::orange': 'an orange back',
    'has_back_color::black': 'a black back',
    'has_back_color::white': 'a white back',
    'has_back_color::red': 'a red back',
    'has_back_color::buff': 'a buff back',
    'has_tail_shape::forked_tail': 'a forked tail',
    'has_tail_shape::rounded_tail': 'a rounded tail',
    'has_tail_shape::notched_tail': 'a notched tail',
    'has_tail_shape::fan-shaped_tail': 'a fan-shaped tail',
    'has_tail_shape::pointed_tail': 'a pointed tail',
    'has_tail_shape::squared_tail': 'a squared tail',
    'has_upper_tail_color::blue': 'a blue upper tail',
    'has_upper_tail_color::brown': 'a brown upper tail',
    'has_upper_tail_color::iridescent': 'an iridescent upper tail',
    'has_upper_tail_color::purple': 'a purple upper tail',
    'has_upper_tail_color::rufous': 'a rufous upper tail',
    'has_upper_tail_color::grey': 'a grey upper tail',
    'has_upper_tail_color::yellow': 'a yellow upper tail',
    'has_upper_tail_color::olive': 'an olive upper tail',
    'has_upper_tail_color::green': 'a green upper tail',
    'has_upper_tail_color::pink': 'a pink upper tail',
    'has_upper_tail_color::orange': 'an orange upper tail',
    'has_upper_tail_color::black': 'a black upper tail',
    'has_upper_tail_color::white': 'a white upper tail',
    'has_upper_tail_color::red': 'a red upper tail',
    'has_upper_tail_color::buff': 'a buff upper tail',
    'has_head_pattern::spotted': 'a spotted head pattern',
    'has_head_pattern::malar': 'a malar head pattern',
    'has_head_pattern::crested': 'a crested head pattern',
    'has_head_pattern::masked': 'a masked head pattern',
    'has_head_pattern::unique_pattern': 'a unique head pattern',
    'has_head_pattern::eyebrow': 'an eyebrow head pattern',
    'has_head_pattern::eyering': 'an eyering head pattern',
    'has_head_pattern::plain': 'a plain head pattern',
    'has_head_pattern::eyeline': 'an eyeline head pattern',
    'has_head_pattern::striped': 'a striped head pattern',
    'has_head_pattern::capped': 'a capped head pattern',
    'has_breast_color::blue': 'a blue breast',
    'has_breast_color::brown': 'a brown breast',
    'has_breast_color::iridescent': 'an iridescent breast',
    'has_breast_color::purple': 'a purple breast',
    'has_breast_color::rufous': 'a rufous breast',
    'has_breast_color::grey': 'a grey breast',
    'has_breast_color::yellow': 'a yellow breast',
    'has_breast_color::olive': 'an olive breast',
    'has_breast_color::green': 'a green breast',
    'has_breast_color::pink': 'a pink breast',
    'has_breast_color::orange': 'an orange breast',
    'has_breast_color::black': 'a black breast',
    'has_breast_color::white': 'a white breast',
    'has_breast_color::red': 'a red breast',
    'has_breast_color::buff': 'a buff breast',
    'has_throat_color::blue': 'a blue throat',
    'has_throat_color::brown': 'a brown throat',
    'has_throat_color::iridescent': 'an iridescent throat',
    'has_throat_color::purple': 'a purple throat',
    'has_throat_color::rufous': 'a rufous throat',
    'has_throat_color::grey': 'a grey throat',
    'has_throat_color::yellow': 'a yellow throat',
    'has_throat_color::olive': 'an olive throat',
    'has_throat_color::green': 'a green throat',
    'has_throat_color::pink': 'a pink throat',
    'has_throat_color::orange': 'an orange throat',
    'has_throat_color::black': 'a black throat',
    'has_throat_color::white': 'a white throat',
    'has_throat_color::red': 'a red throat',
    'has_throat_color::buff': 'a buff throat',
    'has_eye_color::blue': 'blue eyes',
    'has_eye_color::brown': 'brown eyes',
    'has_eye_color::purple': 'purple eyes',
    'has_eye_color::rufous': 'rufous eyes',
    'has_eye_color::grey': 'grey eyes',
    'has_eye_color::yellow': 'yellow eyes',
    'has_eye_color::olive': 'olive eyes',
    'has_eye_color::green': 'green eyes',
    'has_eye_color::pink': 'pink eyes',
    'has_eye_color::orange': 'orange eyes',
    'has_eye_color::black': 'black eyes',
    'has_eye_color::white': 'white eyes',
    'has_eye_color::red': 'red eyes',
    'has_eye_color::buff': 'buff eyes',
    'has_bill_length::about_the_same_as_head': 'a bill about the same length as the head',
    'has_bill_length::longer_than_head': 'a bill longer than the head',
    'has_bill_length::shorter_than_head': 'a bill shorter than the head',
    'has_forehead_color::blue': 'a blue forehead',
    'has_forehead_color::brown': 'a brown forehead',
    'has_forehead_color::iridescent': 'an iridescent forehead',
    'has_forehead_color::purple': 'a purple forehead',
    'has_forehead_color::rufous': 'a rufous forehead',
    'has_forehead_color::grey': 'a grey forehead',
    'has_forehead_color::yellow': 'a yellow forehead',
    'has_forehead_color::olive': 'an olive forehead',
    'has_forehead_color::green': 'a green forehead',
    'has_forehead_color::pink': 'a pink forehead',
    'has_forehead_color::orange': 'an orange forehead',
    'has_forehead_color::black': 'a black forehead',
    'has_forehead_color::white': 'a white forehead',
    'has_forehead_color::red': 'a red forehead',
    'has_forehead_color::buff': 'a buff forehead',
    'has_under_tail_color::blue': 'a blue under tail',
    'has_under_tail_color::brown': 'a brown under tail',
    'has_under_tail_color::iridescent': 'an iridescent under tail',
    'has_under_tail_color::purple': 'a purple under tail',
    'has_under_tail_color::rufous': 'a rufous under tail',
    'has_under_tail_color::grey': 'a grey under tail',
    'has_under_tail_color::yellow': 'a yellow under tail',
    'has_under_tail_color::olive': 'an olive under tail',
    'has_under_tail_color::green': 'a green under tail',
    'has_under_tail_color::pink': 'a pink under tail',
    'has_under_tail_color::orange': 'an orange under tail',
    'has_under_tail_color::black': 'a black under tail',
    'has_under_tail_color::white': 'a white under tail',
    'has_under_tail_color::red': 'a red under tail',
    'has_under_tail_color::buff': 'a buff under tail',
    'has_nape_color::blue': 'a blue nape',
    'has_nape_color::brown': 'a brown nape',
    'has_nape_color::iridescent': 'an iridescent nape',
    'has_nape_color::purple': 'a purple nape',
    'has_nape_color::rufous': 'a rufous nape',
    'has_nape_color::grey': 'a grey nape',
    'has_nape_color::yellow': 'a yellow nape',
    'has_nape_color::olive': 'an olive nape',
    'has_nape_color::green': 'a green nape',
    'has_nape_color::pink': 'a pink nape',
    'has_nape_color::orange': 'an orange nape',
    'has_nape_color::black': 'a black nape',
    'has_nape_color::white': 'a white nape',
    'has_nape_color::red': 'a red nape',
    'has_nape_color::buff': 'a buff nape',
    'has_belly_color::blue': 'a blue belly',
    'has_belly_color::brown': 'a brown belly',
    'has_belly_color::iridescent': 'an iridescent belly',
    'has_belly_color::purple': 'a purple belly',
    'has_belly_color::rufous': 'a rufous belly',
    'has_belly_color::grey': 'a grey belly',
    'has_belly_color::yellow': 'a yellow belly',
    'has_belly_color::olive': 'an olive belly',
    'has_belly_color::green': 'a green belly',
    'has_belly_color::pink': 'a pink belly',
    'has_belly_color::orange': 'an orange belly',
    'has_belly_color::black': 'a black belly',
    'has_belly_color::white': 'a white belly',
    'has_belly_color::red': 'a red belly',
    'has_belly_color::buff': 'a buff belly',
    'has_wing_shape::rounded-wings': 'rounded wings',
    'has_wing_shape::pointed-wings': 'pointed wings',
    'has_wing_shape::broad-wings': 'broad wings',
    'has_wing_shape::tapered-wings': 'tapered wings',
    'has_wing_shape::long-wings': 'long wings',
    'has_size::large_(16_-_32_in)': 'large size (16 - 32 inches)',
    'has_size::small_(5_-_9_in)': 'small size (5 - 9 inches)',
    'has_size::very_large_(32_-_72_in)': 'very large size (32 - 72 inches)',
    'has_size::medium_(9_-_16_in)': 'medium size (9 - 16 inches)',
    'has_size::very_small_(3_-_5_in)': 'very small size (3 - 5 inches)',
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
    'has_back_pattern::solid': 'a solid back pattern',
    'has_back_pattern::spotted': 'a spotted back pattern',
    'has_back_pattern::striped': 'a striped back pattern',
    'has_back_pattern::multi-colored': 'a multi-colored back pattern',
    'has_tail_pattern::solid': 'a solid tail pattern',
    'has_tail_pattern::spotted': 'a spotted tail pattern',
    'has_tail_pattern::striped': 'a striped tail pattern',
    'has_tail_pattern::multi-colored': 'a multi-colored tail pattern',
    'has_belly_pattern::solid': 'a solid belly pattern',
    'has_belly_pattern::spotted': 'a spotted belly pattern',
    'has_belly_pattern::striped': 'a striped belly pattern',
    'has_belly_pattern::multi-colored': 'a multi-colored belly pattern',
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
    'has_wing_pattern::solid': 'a solid wing pattern',
    'has_wing_pattern::spotted': 'a spotted wing pattern',
    'has_wing_pattern::striped': 'a striped wing pattern',
    'has_wing_pattern::multi-colored': 'a multi-colored wing pattern',
}