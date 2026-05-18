splits_2020 = {

    'bloodmnist': [
        {'known': [6, 1, 3, 7], 'unknown': [0, 2, 4, 5]},
        {'known': [7, 2, 1, 6], 'unknown': [4, 3, 0, 5]},
        {'known': [2, 3, 6, 1], 'unknown': [0, 7, 4, 5]},
        {'known': [3, 1, 2, 7], 'unknown': [5, 4, 6, 0]},
        {'known': [0, 7, 4, 6], 'unknown': [1, 3, 5, 2]}
    ],
    'breakhis_40': [
        {'known': [0, 1, 5, 6], 'unknown': [2, 3, 4, 7]},
        {'known': [0, 1, 4, 7], 'unknown': [2, 3, 5, 6]},
        {'known': [1, 3, 6, 7], 'unknown': [0, 2, 4, 5]},
        {'known': [1, 2, 4, 6], 'unknown': [0, 3, 5, 7]},
        {'known': [0, 1, 3, 7], 'unknown': [2, 4, 5, 6]}
    ],
  
    'dermamnist': [
        {'known': [0, 2, 5, 6], 'unknown': [1, 3, 4]},
        {'known': [1, 5, 2, 4], 'unknown': [3, 6, 0]},
        {'known': [5, 1, 6, 4], 'unknown': [0, 3, 2]},
        {'known': [0, 3, 5, 6], 'unknown': [2, 4, 1]}
    ],
    'asc': [
        {'known': [0, 1, 2], 'unknown': [3, 4, 5]},  # Acne, Carcinoma, Eczema vs Keratosis, Milia, Rosacea
        {'known': [0, 3, 4], 'unknown': [1, 2, 5]},  # Acne, Keratosis, Milia vs Carcinoma, Eczema, Rosacea
        {'known': [1, 2, 5], 'unknown': [0, 3, 4]},  # Carcinoma, Eczema, Rosacea vs Acne, Keratosis, Milia
        {'known': [0, 2, 4], 'unknown': [1, 3, 5]}   # Acne, Eczema, Milia vs Carcinoma, Keratosis, Rosacea
    ]
}
"""Skin Conditions: Class Index → Condition Name
0 → Acne
1 → Carcinoma (Skin Cancer)
2 → Eczema  
3 → Keratosis
4 → Milia
5 → Rosacea
"""
