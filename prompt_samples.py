import random

ALLOWED_TYPES = {
    "flat",
    "rough",
    "stairs_up",
    "stairs_down",
    "pyramid",
    "inverse_pyramid",
}

sky = [
    "blue sky",
    "grey sky",
    "cloudy",
    "clouds",
    "sunlight"
    "sun",
    "no sun",
]

material = [
    "concrete",
    "wood",
    "metal",
    "plastic",
    "rubber",
    "glass",
    "ceramic",
    "carpet",
    "cardboard",
    "sandstone",
    "marble",
    "granite",
    "brick",
    "stone",
    "clay",
]

texture = [
    "smooth",
    "rough",
    "bumpy",
    "granular",
    "gravel",
    "sandy",
]

lighting = [
    "dim",
    "dull",
    "neutral lighting",
    "bright",
    "dark",
    "light",
    "shadows",
    "shadowy",
    "shiny"
]


def prompt_gen():
    prompt = ', '.join(
        [random.choice(sky), random.choice(material), random.choice(texture), random.choice(lighting)])
    return prompt
