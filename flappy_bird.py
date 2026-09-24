# Copyright 2026 Rupesh Sreeraman. Licensed under the Apache License, Version 2.0.
# Uses laya-openvino (https://github.com/rupeshs/laya-openvino).
"""Flappy Bird played by laya: one noul question per frame, no generation.
The two details that make the bird fly rather than flap every frame and die in ~50 steps
are in `describe_state` and FLAP_QUESTION.
"""

import argparse
import os
import time

import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0 on import)
import gymnasium

import laya

# The bird is fixed at 20% across the screen; pipes are 52px wide on a 288px screen.
BIRD_X = 0.20
PIPE_WIDTH = 52 / 288


def get_game_state(obs):
    """Pull the bird and its target pipe out of one 12-value observation.

    Layout is `(x, top_y, bottom_y) * 3, bird_y, bird_velocity, bird_rotation`, with the three
    pipes sorted left to right and any pipe still off-screen encoded as the placeholder
    `(1.0, 0.0, 1.0)`. All y values grow *downward*, so a larger `bird_y` means a lower bird.
    """
    pipes = [
        (float(obs[0]), float(obs[1]), float(obs[2])),
        (float(obs[3]), float(obs[4]), float(obs[5])),
        (float(obs[6]), float(obs[7]), float(obs[8])),
    ]

    bird_y = float(obs[9])
    bird_velocity = float(obs[10])

    # Ignore pipes the bird has already cleared.
    available_pipes = [pipe for pipe in pipes if pipe[0] + PIPE_WIDTH >= BIRD_X]

    if available_pipes:
        target_pipe = min(available_pipes, key=lambda pipe: pipe[0])
    else:
        target_pipe = pipes[-1]

    pipe_x, top_y, bottom_y = target_pipe

    gap_center_y = (top_y + bottom_y) / 2

    return {
        "bird_y": bird_y,
        "bird_velocity": bird_velocity,
        "gap_center_y": gap_center_y,
        "offset_from_gap_center": round(bird_y - gap_center_y, 3),
    }


def describe_state(state):
    """Render the state as a sentence rather than a JSON dict.

    laya reads the state as text. Handed the original `{"bird_is_below_gap_center": true, ...}`
    dict it ignores the booleans and answers ~0.9 on every frame, so the bird flaps constantly;
    handed two bare numbers it does no better, because it compares words, not floats. Naming the
    direction in prose is what separates the two cases (mean 0.82 flap vs 0.04 glide).
    """
    offset = state["offset_from_gap_center"]
    return "The bird is flying %s the gap. It sits %.3f %s the gap's center line." % (
        "too low" if offset > 0 else "high enough",
        abs(offset),
        "below" if offset > 0 else "above",
    )


# Note there is no "and is not already rising" clause. The original question carried one, but
# gating on velocity scores 0 pipes; plain below-the-center is the policy that actually flies.
FLAP_QUESTION = {
    "type": "noul",
    "instructions": "The bird is too low and must flap to climb.",
}

FLAP_THRESHOLD = 0.5


def laya_action(agent, obs, latencies, quiet=False):
    state = get_game_state(obs)

    t0 = time.perf_counter()
    response = agent.system_one(describe_state(state), {"should_flap": FLAP_QUESTION})
    latencies.append((time.perf_counter() - t0) * 1000.0)

    answer = response["answers"]["should_flap"]
    probability = answer["noul"]

    if not quiet:
        print(
            "offset: %+.3f  flap probability: %.4f"
            % (state["offset_from_gap_center"], probability)
        )

    if probability >= FLAP_THRESHOLD:
        return 1

    return 0


def build_agent(args):
    if args.torch:
        return laya.load(args.model, subfolder=args.subfolder)
    model_dir = args.ov
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(model_dir)
    return laya.OVAgent(model_dir, device=args.ov_device)


def play(agent, args, seed=None):
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode=None if args.render == "none" else args.render,
        use_lidar=False,
        normalize_obs=True,
    )

    latencies = []
    obs, info = env.reset(seed=seed)

    try:
        while True:
            action = laya_action(agent, obs, latencies, quiet=args.quiet)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        env.close()

    return info["score"], latencies


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ov",
        default="rupeshs/laya-ov-int8",
        metavar="DIR_OR_REPO",
        help="OpenVINO IR directory or Hugging Face repo id (default rupeshs/laya-ov-int8)",
    )
    ap.add_argument(
        "--torch", action="store_true", help="use the torch backend instead of OpenVINO"
    )
    ap.add_argument(
        "--model",
        default="convaiinnovations/laya",
        help="checkpoint for the torch backend",
    )
    ap.add_argument(
        "--subfolder", default=None, help="checkpoint subfolder, e.g. multilingual"
    )
    ap.add_argument("--ov-device", default="CPU", help="OpenVINO device (default CPU)")
    ap.add_argument("--render", default="human", choices=["human", "rgb_array", "none"])
    ap.add_argument("--seeds", type=int, default=1, help="how many episodes to play")
    ap.add_argument("--quiet", action="store_true", help="suppress the per-frame print")
    args = ap.parse_args(argv)

    agent = build_agent(args)

    scores, latencies = [], []
    for i in range(args.seeds):
        score, lat = play(agent, args, seed=i if args.seeds > 1 else None)
        scores.append(score)
        latencies.extend(lat)
        print("Game over. Pipes passed:", score)

    if args.seeds > 1:
        print("Scores: %s | mean %.1f" % (scores, sum(scores) / len(scores)))
    if latencies:
        print(
            "Decisions: %d | mean %.1f ms | max %.1f ms"
            % (len(latencies), sum(latencies) / len(latencies), max(latencies))
        )


if __name__ == "__main__":
    main()
