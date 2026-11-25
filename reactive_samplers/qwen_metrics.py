"""Keep ImageReward's older Transformers dependency in its existing environment."""
import json
import os
from pathlib import Path
import subprocess
import sys


class LegacyMetrics:
    def __init__(self, python, config):
        root = Path(__file__).resolve().parents[1]
        self.process = subprocess.Popen(
            [python, "-u", str(Path(__file__).resolve())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, "PYTHONPATH": str(root)})
        self._request(config)

    def _request(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Quality metric worker stopped; see its stderr in the run log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def score(self, image_path, prompt, reference_image=None):
        return self._request({"image": str(image_path), "prompt": prompt,
                              "reference_image": str(reference_image) if reference_image is not None else None})

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)


def main():
    from contextlib import redirect_stdout
    from reactive_samplers.evaluation import QualityMetrics
    config = json.loads(sys.stdin.readline())
    with redirect_stdout(sys.stderr):
        metrics = QualityMetrics(config['names'], config.get('clip_model'),
                                 config.get('image_reward_checkpoint'), config.get('image_reward_config'))
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        row = json.loads(line)
        with redirect_stdout(sys.stderr):
            result = metrics.score(row['image'], row['prompt'], row.get('reference_image'))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
