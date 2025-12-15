poker-gto is a tiny heads-up holdem sandbox to explore gto-ish self-play.

quickstart
- install python 3.11+
- python -m venv .venv && .venv\\Scripts\\activate
- pip install -r requirements.txt
- python main.py

what you get
- finite betting abstraction with fold/call/three bet sizes/all-in
- deterministic baseline to compare against
- transformer actor-critic with ppo self-play and a rough exploitability probe

notes
- everything is deterministic via seeds inside the env
- rewards are terminal and zero-sum

---
todo:
1. add playing feats vs user
2. add simple pygame design
3. improve the model via risk-aware rl methods
4. destroy all of my friends and win all of their money :)
