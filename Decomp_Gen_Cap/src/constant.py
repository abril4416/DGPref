"""
Some forbidden words and constant variables
"""
import inflect
num_to_word_eng = inflect.engine()

IMAGE_CONSTANT=['scene','image','picture','photo','view','frame','figure','side','backdrop','background']
FORBIDDEN_OBJ=['area','field','ground','grass','glass',
 'checkpoint','city','town','other person', 'precision', 'object','texture','speed','focus', 'right side', 'subject', 'object','remote','side', 'body of water','individual', 'foreground', 'atmosphere', 'waterfront', 'waterside','step','figure','air','skill', 'wall', 'setting','jumping', 'city street', 'side of street', 'surface', 'winter sport','slope','snow-covered slope', 'backdrop','edge of slope','edge','sun', 'winter','design','space', 'detail','arch', 'side of building', 'landmark','element','color','landscape','group','filled', 'with each other','event', 'array','room', 'around table', 'gathering','center of attention','around', 'further away','familiy','size', 'environment', 'above', 'nearby', 'center', 'center of image', 'viewer', 'ambiance','light', 'mode of transportation','transportation','position','direction','assortment', 'center of scene', 'turn', 'each other', 'pattern','style','','street','road','observing','display','','street','corner']

SPATIAL_RELA=[
    "left","right","top","bottom","above","below","near","next","under","beneath","underneath"
]

SCALE=["large","small","huge","big","tiny","long","short","tall","high","larger","smaller"]

FORBIDDEN_ATTR=[
    "multiple","several","various","numerous",
    "middle",'surrounded'
]
FORBIDDEN_ATTR.extend([num_to_word_eng.number_to_words(i) for i in range(2,21)])
FORBIDDEN_ATTR.extend(SPATIAL_RELA)