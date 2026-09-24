# Flappy Bird × Laya (OpenVINO)

Flappy Bird played by [laya-openvino](https://github.com/rupeshs/laya-openvino): each frame asks the model one yes/no (`noul`) question, "the bird is too low and must flap to climb", in a single forward pass with no text generation. It runs on CPU through OpenVINO.

## Setup

```bash
pip install -r requirements.txt
```

On first run the script downloads the int8 OpenVINO IR [`rupeshs/laya-ov-int8`](https://huggingface.co/rupeshs/laya-ov-int8) from Hugging Face and caches it.

## Play

```bash
python flappy_bird.py                                        # watch it play (int8 IR from Hugging Face)
python flappy_bird.py --render none --seeds 3 --quiet        # headless score
```

| Flag | Default | Meaning |
|---|---|---|
| `--ov DIR_OR_REPO` | `rupeshs/laya-ov-int8` | OpenVINO IR directory, or a Hugging Face repo id to download |
| `--ov-device` | `CPU` | OpenVINO device |
| `--torch` | off | use the torch backend (`--model`, default `convaiinnovations/laya`) |
| `--render` | `human` | `human`, `rgb_array` or `none` |
| `--seeds N` | `1` | number of games to play |
| `--quiet` | off | hide the per-frame flap probability |

A headless run on CPU with the int8 IR passed 14, 20 and 58 pipes over three seeds (mean 30.7), at a mean 28 ms per decision (the fp16 IR scores the same at 94 ms). Scores vary with the seed.

## How it works

The observation is reduced to one number, the bird's offset from the next gap's center line, and described in a sentence (`describe_state`). That wording matters: given a JSON dict or bare numbers, the model answers about 0.9 on every frame and the bird flaps itself into the ceiling. Named in prose ("too low" / "high enough"), the flap probability separates cleanly (mean 0.82 vs 0.04). If the probability is 0.5 or higher, the bird flaps.

## License

Apache 2.0, see [LICENSE](LICENSE). Laya is developed by Convai Innovations ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)); this project is not affiliated with them. [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) is MIT-licensed.
