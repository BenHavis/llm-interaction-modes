# LLM Interaction Modes Experiment

An experiment in explicit interaction modes for LLMs. Explores using user-selected behavioral modes as explicit system-level contracts, instead of relying on the model to infer user intent from conversational cues.

## Motivation

When interacting with LLMs, users often want different types of engagement—brainstorming, critical review, collaborative problem-solving—but rely on the model to infer intent from conversational cues. This experiment tests whether explicit mode declaration produces more predictable, controllable behavior than implicit inference.

## Modes

| Mode | Role | Behavior |
|------|------|----------|
| `default` | Helpful assistant | Clear, accurate, neutral responses |
| `collaboration` | Partner | Builds on ideas, offers suggestions, avoids criticism |
| `brainstorm` | Creative generator | Many ideas quickly, no evaluation, quantity over quality |
| `critique` | Critical reviewer | Identifies weaknesses, challenges assumptions, no hedging |

Each mode operates as a behavioral contract: the model commits to specific behaviors and suppresses others.

## Setup

```bash
# Install dependencies
pip install langchain-anthropic python-dotenv

# Configure API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Usage

Run `main.py` from your terminal (`python main.py`) or directly from your IDE.

### Commands

| Command | Description |
|---------|-------------|
| `/mode <name>` | Switch to a mode (default, collaboration, brainstorm, critique) |
| `/mode` | Show current mode |
| `/clear` | Clear conversation history |
| `/quit` | Exit (shows violation summary) |

## Features

**State awareness**: Each mode includes self-knowledge—the model can report which mode it's in and explain its behavior in terms of that mode's contract.

**Mode validation**: Brainstorm mode tracks "contract violations"—responses that include critique markers (`however`, `but`, `risk`, `problem`, etc.) when the contract specifies no evaluation. Violations are summarized on exit.

**Persistent history**: Conversation history persists across mode switches, allowing workflows like "brainstorm in one mode, then switch to critique mode to evaluate the results."

## Example

```
You: /mode brainstorm
Mode set to: brainstorm

You: Ways to improve onboarding for a developer tool

Claude:
[BRAINSTORM MODE]
- Interactive sandbox that works before signup
- "Time to first deploy" as the core metric
- Video walkthroughs with copy-paste commands
- CLI that generates starter projects
- Onboarding checklist with progress bar
- Slack community invite during signup
- Templates for common use cases
- "Fork this repo" instead of documentation
- AI assistant embedded in the docs
- Weekly office hours for new users
...
```

## Research Questions

- Do explicit modes produce meaningfully different outputs compared to inline instructions ("please brainstorm this")?
- How strongly do mode contracts hold when users request conflicting behavior?
- Does mode-based interaction improve perceived control, actual outcomes, or both?

## Built With

- [LangChain](https://github.com/langchain-ai/langchain)
- [Claude](https://www.anthropic.com/claude) (Anthropic)