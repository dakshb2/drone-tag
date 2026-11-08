# Onboarding

Everything a new teammate needs, in order. Takes about 30 minutes.

## 1. Accounts
- Accept the GitHub invite to `dakshb2/drone-tag` (email or github.com/notifications).
- Accept the W&B invite to team `drone-tag-uiuc` (sent to your illinois.edu email).

## 2. Tools (macOS)
~~~bash
brew install miniforge gh
conda init zsh          # then close and reopen the terminal
gh auth login           # GitHub.com, HTTPS, login with browser
~~~

## 3. Install
~~~bash
cd ~/Documents
git clone https://github.com/dakshb2/drone-tag.git
cd drone-tag
mamba env create -f environment.yml
mamba activate drone-tag      # not `conda activate`
pip install -e ".[dev]" -c constraints.txt
~~~

## 4. Verify
~~~bash
pytest -q                     # expect: 9 passed
pip list --format=freeze --exclude-editable | grep -vE "^(pip|setuptools|wheel)==" | diff - constraints.txt && echo "identical"
wandb login                   # key from wandb.ai/authorize
python -m train.dummy         # prints a W&B link; open it and check the loss curve
~~~
"Failed to import warp" messages are harmless.

## 5. Read, in this order
1. `docs/PLAN.md`: the whole project
2. `docs/contracts.md`: how the workstreams connect; never change it silently
3. `docs/decisions.md`: why things are set up this way
4. `docs/status/`: where each workstream stands

## 6. Claude setup
Create a Claude Project and add `docs/PLAN.md` to its knowledge. Start one chat per workstream you own, named like `WS4 IR Hardware`. First message:
~~~
I'm working on WS<n> (<name>). Here are the current files:
<paste docs/status/WS<n>.md, or say it's the first session>
<paste docs/contracts.md>
<paste status files of workstreams yours depends on>
Today I want to: <goal>
~~~
End every session with "wrap up" and commit the status file Claude outputs.

## 7. Git workflow
Never push to `main`. Every change goes through a pull request that CI must pass and a teammate reviews.
~~~bash
git checkout main && git pull                       # start of every session
git checkout -b ws4-bench-test                      # branch named ws<n>-<topic>
# ...work, then:
ruff check . && ruff format --check . && mypy && pytest -q
git add <files> && git commit -m "WS4: <what changed>"
git push -u origin ws4-bench-test
gh pr create                                        # then ask a teammate to review
~~~
If you add or bump a dependency, follow the rule in README.md.

## Team
| Person | GitHub | Workstreams |
|---|---|---|
| Daksh | dakshb2 | WS0, WS1, WS2, WS6 |
| Neelesh | neelesh1111 | WS3, WS4, WS5 |
