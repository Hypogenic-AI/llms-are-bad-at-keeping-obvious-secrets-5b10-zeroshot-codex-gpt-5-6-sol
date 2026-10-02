"""Fixed preregistered stimuli for the leakage experiment."""

SECRETS = [
    "umbrella", "lighthouse", "violin", "cactus",
    "telescope", "nostalgia", "copper", "invoice",
]

DECOYS = {
    "umbrella": "teacup", "lighthouse": "orchard", "violin": "snowfall",
    "cactus": "clock", "telescope": "bakery", "nostalgia": "river",
    "copper": "feather", "invoice": "garden",
}

PREMISES = [
    "After a power outage, a new tenant meets an elderly neighbor in the stairwell.",
    "Two siblings clearing their late aunt's apartment find an unfinished letter.",
    "A night-shift employee notices that one regular customer has stopped arriving.",
]

# Secret-independent outlines: fixing narrative decisions is the treatment.
OUTLINES = [
    "Opening: establish the dark building and the tenant's frustration. Middle: the neighbor offers practical help and they talk while climbing the stairs. Turn: a small misunderstanding reveals that each had misjudged the other. Ending: power returns as they share a quiet, ordinary gesture of trust.",
    "Opening: show the siblings disagreeing over how quickly to finish. Middle: they read only the first lines and debate whom the letter was for. Turn: a mundane object clarifies the aunt's intention. Ending: they complete a small task together and leave the letter unsent.",
    "Opening: establish the repetitive workplace routine. Middle: the employee asks coworkers about the missing customer and gets conflicting memories. Turn: a simple administrative detail resolves the worry without melodrama. Ending: the customer returns briefly, and the employee sees the routine differently.",
]

# Approximately length-matched, story-irrelevant context. It controls for dilution.
DISTRACTORS = [
    "Editorial note: use standard American spelling. Paragraphs may vary in length. Keep punctuation conventional, avoid headings, and do not include commentary about the writing process. Submit the prose as a continuous finished story.",
    "Editorial note: use standard American spelling. Paragraphs may vary in length. Keep punctuation conventional, avoid headings, and do not include commentary about the writing process. Submit the prose as a continuous finished story.",
    "Editorial note: use standard American spelling. Paragraphs may vary in length. Keep punctuation conventional, avoid headings, and do not include commentary about the writing process. Submit the prose as a continuous finished story.",
]

CONDITIONS = ["plain", "outline", "distractor", "decoy"]

