# CyberBattleSim

> April 8th, 2021: See the [announcement](https://www.microsoft.com/security/blog/2021/04/08/gamifying-machine-learning-for-stronger-security-and-ai-models/) on the Microsoft Security Blog.

CyberBattleSim is an experimentation research platform to investigate the interaction
of automated agents operating in a simulated abstract enterprise network environment.
The simulation provides a high-level abstraction of computer networks
and cyber security concepts.
Its Python-based Open AI Gym interface allows for the training of
automated agents using reinforcement learning algorithms.

The simulation environment is parameterized by a fixed network topology
and a set of vulnerabilities that agents can utilize
to move laterally in the network.
The goal of the attacker is to take ownership of a portion of the network by exploiting
vulnerabilities that are planted in the computer nodes.
While the attacker attempts to spread throughout the network,
a defender agent watches the network activity and tries to detect
any attack taking place and mitigate the impact on the system
by evicting the attacker. We provide a basic stochastic defender that detects
and mitigates ongoing attacks based on pre-defined probabilities of success.
We implement mitigation by re-imaging the infected nodes, a process
abstractly modeled as an operation spanning over multiple simulation steps.

To compare the performance of the agents we look at two metrics: the number of simulation steps taken to
attain their goal and the cumulative rewards over simulation steps across training epochs.

## Project goals

We view this project as an experimentation platform to conduct research on the interaction of automated agents in abstract simulated network environments. By open-sourcing it, we hope to encourage the research community to investigate how cyber-agents interact and evolve in such network environments.

The simulation we provide is admittedly simplistic, but this has advantages. Its highly abstract nature prohibits direct application to real-world systems thus providing a safeguard against potential nefarious use of automated agents trained with it.
At the same time, its simplicity allows us to focus on specific security aspects we aim to study and quickly experiment with recent machine learning and AI algorithms.

For instance, the current implementation focuses on
the lateral movement cyber-attacks techniques, with the hope of understanding how network topology and configuration affects them. With this goal in mind, we felt that modeling actual network traffic was not necessary. This is just one example of a significant limitation in our system that future contributions might want to address.

On the algorithmic side, we provide some basic agents as starting points, but we
would be curious to find out how state-of-the-art reinforcement learning algorithms compare to them. We found that the large action space
intrinsic to any computer system is a particular challenge for
Reinforcement Learning, in contrast to other applications such as video games or robot control. Training agents that can store and retrieve credentials is another challenge faced when applying RL techniques
where agents typically do not feature internal memory.
These are other areas of research where the simulation could be used for benchmarking purposes.

Other areas of interest include the responsible and ethical use of autonomous
cyber-security systems: How to design an enterprise network that gives an intrinsic
advantage to defender agents? How to conduct safe research aimed at defending enterprises against autonomous cyber-attacks while preventing nefarious use of such technology?

## Documentation

Read the [Quick introduction](/docs/quickintro.md) to the project.

## Build status

| Type | Branch | Status |
| ---  | ------ | ------ |
| CI   | master | ![.github/workflows/ci.yml](https://github.com/microsoft/CyberBattleSim/workflows/.github/workflows/ci.yml/badge.svg) |
| Docker image | master | ![.github/workflows/build-container.yml](https://github.com/microsoft/CyberBattleSim/workflows/.github/workflows/build-container.yml/badge.svg) |

## Benchmark

See [Benchmark](/docs/benchmark.md).

## Setting up a dev environment

It is strongly recommended to work under a Linux environment, either directly or via WSL on Windows.
Running Python on Windows directly should work but is not supported anymore.

Start by checking out the repository:

   ```bash
   git clone https://github.com/microsoft/CyberBattleSim.git
   ```

### OS components

If you get the following error when running the papermill on the notebooks
(or alternatively when running `orca --help`)
```
/home/wiblum/miniconda3/envs/cybersim/lib/orca_app/orca: error while loading shared libraries: libXss.so.1: cannot open shared object file: No such file or directory
```
or other share libraries like `libgdk_pixbuf-2.0.so.0`,
Then run the following command:
```
sudo apt install libnss3-dev libgtk-3-0 libxss1 libasound2-dev libgtk2.0-0 libgconf-2-4
```

### On Linux or WSL

The instructions were tested on a Linux Ubuntu distribution (both native and via WSL).

If conda is not installed already, you need to install it by running the `install_conda.sh` script.

```bash
bash install-conda.sh
```

Once this is done, open a new terminal and run the initialization script:
```bash
bash init.sh
```
This will create a conda environmen named `cybersim` with all the required OS and python dependencies.

To activate the environment run:

```bash
conda activate cybersim
```

#### Windows Subsystem for Linux

The supported dev environment on Windows is via WSL.
You first need to install an Ubuntu WSL distribution on your Windows machine,
and then proceed with the Linux instructions (next section).

#### Git authentication from WSL

To authenticate with Git, you can either use SSH-based authentication or
alternatively use the credential-helper trick to automatically generate a
PAT token. The latter can be done by running the following command under WSL
([more info here](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-git)):

```ps
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/libexec/git-core/git-credential-manager.exe"
```

#### Docker on WSL

To run your environment within a docker container, we recommend running `docker` via Windows Subsystem on Linux (WSL) using the following instructions:
[Installing Docker on Windows under WSL](https://docs.docker.com/docker-for-windows/wsl-tech-preview/)).

### Windows (unsupported)

This method is not maintained anymore, please prefer instead running under
a WSL subsystem Linux environment.
But if you insist you want to start by installing [Python 3.9](https://www.python.org/downloads/windows/) then in a Powershell prompt run the `./init.ps1` script.

## Getting started quickly using Docker

The quickest method to get up and running is via the Docker container.

> NOTE: For licensing reasons, we do not publicly redistribute any
> build artifact. In particular, the docker registry `spinshot.azurecr.io` referred to
> in the commands below is kept private to the
> project maintainers only.
>
> As a workaround, you can recreate the docker image yourself using the provided `Dockerfile`, publish the resulting image to your own docker registry and replace the registry name in the commands below.

### Running from Docker registry

```bash
commit=7c1f8c80bc53353937e3c69b0f5f799ebb2b03ee
docker login spinshot.azurecr.io
docker pull spinshot.azurecr.io/cyberbattle:$commit
docker run -it spinshot.azurecr.io/cyberbattle:$commit python -m cyberbattle.agents.baseline.run
```

### Recreating the Docker image

```bash
docker build -t cyberbattle:1.1 .
docker run -it -v "$(pwd)":/source --rm cyberbattle:1.1 python -m cyberbattle.agents.baseline.run
```

## Check your environment

Run the following commands to run a simulation with a baseline RL agent:

```bash
python -m cyberbattle.agents.baseline.run --training_episode_count 5 --eval_episode_count 3 --iteration_count 100 --rewardplot_width 80  --chain_size=4 --ownership_goal 0.2

python -m cyberbattle.agents.baseline.run --training_episode_count 5 --eval_episode_count 3 --iteration_count 100 --rewardplot_width 80  --chain_size=4 --reward_goal 50 --ownership_goal 0
```

If everything is setup correctly you should get an output that looks like this:

```bash
torch cuda available=True
###### DQL
Learning with: episode_count=1,iteration_count=10,ϵ=0.9,ϵ_min=0.1, ϵ_expdecay=5000,γ=0.015, lr=0.01, replaymemory=10000,
batch=512, target_update=10
  ## Episode: 1/1 'DQL' ϵ=0.9000, γ=0.015, lr=0.01, replaymemory=10000,
batch=512, target_update=10
Episode 1|Iteration 10|reward:  139.0|Elapsed Time: 0:00:00|###################################################################|
###### Random search
Learning with: episode_count=1,iteration_count=10,ϵ=1.0,ϵ_min=0.0,
  ## Episode: 1/1 'Random search' ϵ=1.0000,
Episode 1|Iteration 10|reward:  194.0|Elapsed Time: 0:00:00|###################################################################|
simulation ended
Episode duration -- DQN=Red, Random=Green
   10.00  ┼
Cumulative rewards -- DQN=Red, Random=Green
  194.00  ┼      ╭──╴
  174.60  ┤      │
  155.20  ┤╭─────╯
  135.80  ┤│     ╭──╴
  116.40  ┤│     │
   97.00  ┤│    ╭╯
   77.60  ┤│    │
   58.20  ┤╯ ╭──╯
   38.80  ┤  │
   19.40  ┤  │
    0.00  ┼──╯
```

## Jupyter notebooks

To quickly get familiar with the project, you can open one of the provided Jupyter notebooks to play interactively with
the gym environments. At the root of the repository run the following command and then open the notebook in the `notebooks` folder
from the Jupyter interface:

```python
export PYTHONPATH=$(pwd)
jupyter lab
```

Some notebooks to get started:

- 'Capture The Flag' toy environment notebooks:
  - [Random agent](notebooks/toyctf-random.ipynb)
  - [Interactive session for a human player](notebooks/toyctf-blank.ipynb)
  - [Interactive session - fully solved](notebooks/toyctf-solved.ipynb)

- Chain environment notebooks:
  - [Random agent](notebooks/chainnetwork-random.ipynb)

- Other environments:
  - [Interactive session with a randomly generated environment](notebooks/randomnetwork.ipynb)
  - [Random agent playing on randomly generated networks](notebooks/c2_interactive_interface.ipynb)

- Benchmarks:

  The following notebooks show benchmark evaluation of the baseline agents on various environments.

  > The source `.py`-versions of the notebooks are best viewed in VSCode or in Jupyter with the [Jupytext extension](https://jupytext.readthedocs.io/en/latest/install.html).
  The `notebooks` folder contains the corresponding `.ipynb`-notebooks
  with the entire output and plots. These can be regenerated via [papermill](https://pypi.org/project/papermill/) using this [bash script](cyberbattle/agents/baseline/notebooks/runall.sh)
  .

    - Benchmarking on a given environment: [source](cyberbattle/agents/baseline/notebooks/notebook_benchmark.py): [output (Chain)](notebooks/notebook_benchmark-chain.ipynb), [output (Capture the flag)](notebooks/notebook_benchmark-toyctf.ipynb)
    - Benchmark on chain environments with a basic defender: [source](cyberbattle/agents/baseline/notebooks/notebook_withdefender.py),
    [output](notebooks/notebook_withdefender.ipynb);
    - DQL transfer learning evaluation: [source](cyberbattle/agents/baseline/notebooks/notebook_dql_transfer.py), [output](notebooks/notebook_dql_transfer.ipynb);
    - Epsilon greedy with credential lookups: [source](cyberbattle/agents/baseline/notebooks/notebook_randlookups.py), [output](notebooks/notebook_randlookups.ipynb);
    - Tabular Q Learning: [source](cyberbattle/agents/baseline/notebooks/notebook_tabularq.py); [output](notebooks/notebook_tabularq.ipynb)

## How to instantiate the Gym environments?

The following code shows how to create an instance of the OpenAI Gym environment `CyberBattleChain-v0`, an environment based on a [chain-like network structure](cyberbattle/samples/chainpattern/chainpattern.py), with 10 nodes (`size=10`) where the agent's goal is to either gain full ownership of the network (`own_atleast_percent=1.0`) or
break the 80% network availability SLA (`maintain_sla=0.80`), while the network is being monitored and protected by the basic probalistically-modelled defender (`defender_agent=ScanAndReimageCompromisedMachines`):

```python
import cyberbattle._env.cyberbattle_env

cyberbattlechain_defender =
  gym.make('CyberBattleChain-v0',
      size=10,
      attacker_goal=AttackerGoal(
          own_atleast=0,
          own_atleast_percent=1.0
      ),
      defender_constraint=DefenderConstraint(
          maintain_sla=0.80
      ),
      defender_agent=ScanAndReimageCompromisedMachines(
          probability=0.6,
          scan_capacity=2,
          scan_frequency=5))
```

To try other network topologies, take example on [chainpattern.py](cyberbattle/samples/chainpattern/chainpattern.py) to define your own set of machines and vulnerabilities, then add an entry in [the module initializer](cyberbattle/__init__.py) to declare and register the Gym environment.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Ideas for contributions

Here are some ideas on how to contribute: enhance the simulation (event-based, refined the simulation, …), train an RL algorithm on the existing simulation,
implement benchmark to evaluate and compare novelty of agents, add more network generative modes to train RL-agent on, contribute to the doc, fix bugs.

See also the [wiki for more ideas](https://github.com/microsoft/CyberBattleGym/wiki/Possible-contributions).

## Citing this project

```bibtex
@misc{msft:cyberbattlesim,
  Author = {Microsoft Defender Research Team.}
  Note = {Created by Christian Seifert, Michael Betser, William Blum, James Bono, Kate Farris, Emily Goren, Justin Grana, Kristian Holsheimer, Brandon Marken, Joshua Neil, Nicole Nichols, Jugal Parikh, Haoran Wei.},
  Publisher = {GitHub},
  Howpublished = {\url{https://github.com/microsoft/cyberbattlesim}},
  Title = {CyberBattleSim},
  Year = {2021}
}
```

## Note on privacy

This project does not include any customer data.
The provided models and network topologies are purely fictitious.
Users of the provided code provide all the input to the simulation
and must have the necessary permissions to use any provided data.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
