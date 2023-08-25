default_config = {
    # Preprocessing
    "min_combined_length": 10,
    "to_lower": True,
    "to_upper": False,
    "remove_punctuation": True,
    "remove_chinese": True,
    "stem": False,
    "only_4digit": True,
    "only_exist": True,

    # Data
    "add_isco88": True,

    # Embeddings
    "embeddings_engine": "fasttext",

    # Final Model
    "model": "knn",  # knn, gradient_boosting
}
