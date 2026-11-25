# AutoCache: Agent-Designed Caching for Image and Video Generation

[🌐 Project website](https://vibe-hust.github.io/autocache/) · [💻 Code repository](https://github.com/vibe-hust/autocache_code)

AutoCache uses an agent to discover executable caching policies for frozen image
and video generators. This repository contains the search controllers, model
adapters, baseline implementations, evaluation code, prompt/seed manifests, and
frozen policies from the prepared paper code release.

<a id="setup-and-cpu-configuration-checks"></a>

## 🛠️ Setup and CPU configuration checks

Use Python 3.10 or newer and run commands from the repository root:

```bash
git clone https://github.com/vibe-hust/autocache_code.git
cd autocache_code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m variants.opal_search.search check experiments/qwen21-drawbench30-170/search.json
python -m variants.opal_search.search check experiments/flux3-drawbench30-170/search.json
python -m variants.wan21_opal.search check experiments/wan21-vbench124-820/search.json
```

The `check` commands validate configurations without generating samples or
calling the agent. Installing this package does not install the full GPU model
stack. Generation additionally needs the dependencies for the chosen backbone,
its model weights, and the metric checkpoints listed in the generation config.
See the bundled baseline requirements and [metric setup](metrics/README.md).

<a id="components"></a>

## 🗂️ Components

| Folder | Contents |
| --- | --- |
| `variants/opal_search/` | Qwen-Image-2.1 and FLUX3-Image-Klein-Base policy search |
| `variants/wan21_opal/` | Wan2.1-T2V-1.3B policy search |
| `reactive_samplers/` | Model adapters, observations, tracing, and evaluation |
| `experiments/` | Search settings and fixed prompt/seed manifests |
| `policies/` | Frozen `signals.py` and `updater.py` policy pairs |
| `baselines/` | Baseline implementations and retained upstream sources |
| `metrics/` | Metric support code and bundled CLIP source |
| `scripts/` | Latency and operator-count measurement utilities |
| `site/trento/` | Optional Slurm submission and remote agent relay tools |

<a id="search-and-evaluation"></a>

## 📊 Search and evaluation

The image and video controllers expose `check`, `run`, and `test`. Read the
[image search guide](variants/opal_search/README.md) or
[video search guide](variants/wan21_opal/README.md) before a GPU run. Configure
model and metric paths in the corresponding `generation.json` files. Preserve
the supplied prompt splits, seeds, and sampling settings for paper comparisons.

The supplied paper search configurations select the optional remote agent
relay through `agent.command`. That relay requires an operator-configured SSH
host, CLI installation, and existing authentication. For a local agent, remove
`agent.command` to use the built-in `agent_cli.py`, and configure `agent.model`,
`agent.profile`, and `agent.reasoning_effort` for your own installation. The
relay worker's options are documented by `site/trento/bridge_worker.py --help`.
Credentials and authentication state are not included in this repository.

Frozen policies can be inspected in `policies/`; new searches and evaluations
write separate run directories. Model checkpoints and generated outputs must
be downloaded or produced separately.

<a id="license"></a>

## 📄 License

The supplied project license is [MIT](LICENSE). Bundled third-party components
retain their own copyright notices and licenses in their respective folders.
