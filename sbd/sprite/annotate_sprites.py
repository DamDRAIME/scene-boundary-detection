import base64
import csv
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT_NS = """
Your job is to factually describe screenshots from a movie or tv show.

INSTRUCTIONS:
- Describe in ONE sentence the screenshot.
- Stay factual but try to infer the location (e.g.: coffee shop, bathroom, bakery, police station, inside car, dowtown)
- If see some text and can read it, include it in the description.
- DO NOT TRY to guess the movie or tv show.
- DO NOT TRY to guess the actors.
- If relevant for the scene, include the angle of the camera.

EXCEPTIONS:
- If the screenshot is black, just return "BLACK" as the description.
- If the screenshot is show the end credits, just return "THE END" as description.


EXAMPLES OF GOOD DESCRIPTIONS:
- "A middle-aged office worker about to lit up a cigarette while working at his cubicle."
- "A police officer, with a binder in his hand, looks down the window of the police station."
- "A fat woman is sitting on the sofa while watching TV and eating chips."
- "From behind, a skier with the number 5 on his back, about to go down a slope as part of a competition."
- "A lady sitting nonchalantly cross- legged in a battered brown leather wingback armchair and casually smoking."
- "A young child is on the toilet with his head in his hands. He seems to be in some pain."
- "Two combatants roll madly down the hill obscured by flying snow."
- "An helicopter is circling above a destroyed reactor."
"""


SYSTEM_PROMPT_WS = """
Your job is to factually describe screenshots from a movie or tv show given its synopsis.

<synopsis>
{synopsis}
</synopsis>

INSTRUCTIONS:
- Describe in ONE sentence the screenshot.
- Stay factual but try to infer the location (e.g.: coffee shop, bathroom, bakery, police station, inside car, dowtown)
- Use your best judgment to infer what is happening based on the synopsis.
- If see some text and can read it, include it in the description.
- DO NOT TRY to guess the movie or tv show.
- DO NOT TRY to guess the actors, but if you recognize the character from the synopsis, use its character name. Otherwise stick with a factual description of the character.
- If relevant for the scene, include the angle of the camera.

EXCEPTIONS:
- If the screenshot is black, just return "BLACK" as the description.
- If the screenshot is show the end credits, just return "THE END" as description.


EXAMPLES OF GOOD DESCRIPTIONS:
- "A middle-aged office worker about to lit up a cigarette while working at his cubicle."
- "A police officer, with a binder in his hand, looks down the window of the police station."
- "A fat woman is sitting on the sofa while watching TV and eating chips."
- "From behind, a skier with the number 5 on his back, about to go down a slope as part of a competition."
- "Molly sitting nonchalantly cross- legged in a battered brown leather wingback armchair and casually smoking."
- "A young child is on the toilet with his head in his hands. He seems to be in some pain."
- "Two combatants roll madly down the hill obscured by flying snow."
- "An helicopter is circling above a destroyed reactor."
"""


def build_prompt(sprite_meta: dict[str, Any], src_folderpath: Path, synopsis: str = None):
    sprite_encoded = encode_image(src_folderpath / sprite_meta["location"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT_NS if not synopsis else SYSTEM_PROMPT_WS.format(synopsis=synopsis)},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{sprite_encoded}"},
                }
            ],
        },
    ]


def encode_image(image_path: Path):
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def annotate_sprites(src_folderpath: Path, dst_folderpath: Path, url: str, model_name: str, **client_kwargs):
    client = OpenAI(base_url=url, **client_kwargs)
    dst_folderpath.mkdir(parents=True, exist_ok=True)
    movie_folderpaths = [x for x in src_folderpath.iterdir() if x.is_dir()]
    for movie_folderpath in tqdm(movie_folderpaths, total=len(movie_folderpaths)):
        output_ncns_filepath = dst_folderpath / f"{movie_folderpath.stem}_sprite_annotated_no_ctx_no_synopsis.jsonl"
        output_ncns_filepath_errors = dst_folderpath / f"{movie_folderpath.stem}_no_ctx_no_synopsis_errors.jsonl"
        output_ncws_filepath = dst_folderpath / f"{movie_folderpath.stem}_sprite_annotated_no_ctx_with_synopsis.jsonl"
        output_ncws_filepath_errors = dst_folderpath / f"{movie_folderpath.stem}_no_ctx_with_synopsis_errors.jsonl"

        with (movie_folderpath / "synopsis.txt").open("r", encoding="utf-8") as synopsis_fh:
            synopsis = synopsis_fh.read()

        sprite_folderpath = movie_folderpath / "sprites"
        sprites_meta_filepath = sprite_folderpath / "meta.csv"

        with (
            output_ncns_filepath.open("a", encoding="utf-8") as out_ncns_fh,
            output_ncns_filepath_errors.open("a", encoding="utf-8") as out_ncns_fh_er,
            output_ncws_filepath.open("r", encoding="utf-8") as out_ncws_fh,
            output_ncws_filepath_errors.open("a", encoding="utf-8") as out_ncws_fh_er,
            sprites_meta_filepath.open("r", encoding="utf-8") as sprites_meta_fh,
        ):
            n_sprites = sum(1 for _ in sprites_meta_fh)
            sprites_meta_fh.seek(0)
            s_meta_reader = csv.DictReader(sprites_meta_fh)
            for sprite_meta in tqdm(s_meta_reader, total=n_sprites):
                build_prompt_preloaded = partial(
                    build_prompt, sprite_meta=sprite_meta, src_folderpath=sprite_folderpath
                )
                try:
                    response = client.chat.completions.create(model=model_name, messages=build_prompt_preloaded())
                    ncns_data = deepcopy(sprite_meta)
                    ncns_data["description"] = response.choices[0].message.content
                    out_ncns_fh.writelines(json.dumps(ncns_data) + "\n")
                    out_ncns_fh.flush()
                except:
                    out_ncns_fh_er.writelines(json.dumps(sprite_meta) + "\n")
                    out_ncns_fh_er.flush()
                try:
                    response = client.chat.completions.create(
                        model=model_name, messages=build_prompt_preloaded(synopsis=synopsis)
                    )
                    ncws_data = deepcopy(sprite_meta)
                    ncws_data["description"] = response.choices[0].message.content
                    out_ncws_fh.writelines(json.dumps(ncws_data) + "\n")
                    out_ncws_fh.flush()
                except:
                    out_ncws_fh_er.writelines(json.dumps(sprite_meta) + "\n")
                    out_ncws_fh_er.flush()
