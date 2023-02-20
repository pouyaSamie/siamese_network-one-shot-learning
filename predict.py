def predict(model, image_path, train_data):
    # Load the query image
    query_image = load_image(image_path)

    # Encode the query image
    query_embedding = model.encode(query_image)

    # Find the most similar images in the training set
    similarities = []
    for i in range(train_data.num_classes):
        class_embeddings = train_data.get_embeddings_for_class(i)
        for j in range(train_data.num_support):
            support_embedding = class_embeddings[j]
            similarity = cosine_similarity(query_embedding, support_embedding)
            similarities.append((i, j, similarity))
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Get the top result
    top_result = similarities[0]
    class_index = top_result[0]
    support_index = top_result[1]
    class_name = train_data.class_names[class_index]
    support_image_path = train_data.get_image_path(class_index, support_index)

    return class_name, support_image_path
