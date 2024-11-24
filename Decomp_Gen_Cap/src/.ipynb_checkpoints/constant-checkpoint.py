"""
Some forbidden words and constant variables
"""
import inflect
num_to_word_eng = inflect.engine()

IMAGE_CONSTANT=['scene','image','picture','photo','view','frame']

SPATIAL_RELA=[
    "left","right","top","bottom","above","below","near","next","under","beneath","underneath"
]

SCALE=["large","small","huge","big","tiny","long","short","tall","high","larger","smaller"]

FORBIDDEN_ATTR=[
    "multiple","several","various","numerous",
    "middle"
]
FORBIDDEN_ATTR.extend([num_to_word_eng.number_to_words(i) for i in range(2,21)])
FORBIDDEN_ATTR.extend(SPATIAL_RELA)