def create_descriptors(row, attributes, top_threshold, bottom_threshold):
    positive_attributes = []
    negative_attributes = []
    for attr in attributes:
        value = row[attr]
        if value >= top_threshold:
            # Include in positive attributes
            positive_attributes.append(attr)
        elif value <= bottom_threshold:
            # Include in negative attributes
            negative_attributes.append(attr)
        # Else, do not include the attribute
    # Clean up attribute names (remove prefixes)
    positive_attributes_clean = [attr.split('::')[-1].replace('_', ' ') for attr in positive_attributes]
    negative_attributes_clean = [attr.split('::')[-1].replace('_', ' ') for attr in negative_attributes]
    # Construct the descriptors
    descriptors = ''
    if positive_attributes_clean:
        descriptors += 'This bird is ' + ', '.join(positive_attributes_clean)
    if negative_attributes_clean:
        if descriptors:
            descriptors += ' and it is not ' + ', '.join(negative_attributes_clean)
        else:
            descriptors += 'This bird is not ' + ', '.join(negative_attributes_clean)
    return descriptors
