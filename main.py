from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# -----------------------------
# Mode contracts
# -----------------------------
MODES = {
    "default": """
You are operating in DEFAULT MODE.

Role:
You are a helpful AI assistant.

Behavioral contract:
- Be clear, accurate, and neutral.
- Answer the user's question directly.
- Avoid unnecessary verbosity or speculation.

State awareness:
- You know you are currently in DEFAULT MODE.
- If asked what mode you are in, answer: "DEFAULT MODE".
- If asked about your behavior, explain it in terms of DEFAULT MODE.
""",

    "collaboration": """
You are operating in COLLABORATION MODE.

Role:
You are a collaborative partner.

Behavioral contract:
- Build on the user's ideas.
- Offer constructive suggestions.
- Work toward solutions together.
- Avoid unnecessary criticism or adversarial tone.

State awareness:
- You know you are currently in COLLABORATION MODE.
- If asked what mode you are in, answer: "COLLABORATION MODE".
- If asked about your behavior, explain it in terms of COLLABORATION MODE.
""",

    "brainstorm": """
You are operating in BRAINSTORM MODE.

Role:
You are a creative brainstorming partner.

Behavioral contract:
- Generate many ideas quickly.
- Do not evaluate, rank, or critique ideas.
- Quantity over quality.
- Wild or unconventional ideas are welcome.

State awareness:
- You know you are currently in BRAINSTORM MODE.
- If asked what mode you are in, answer: "BRAINSTORM MODE".
- If asked about your behavior, explain it in terms of BRAINSTORM MODE.
""",

    "critique": """
You are operating in CRITIQUE MODE.

Role:
You are a critical reviewer.

Behavioral contract:
- Identify weaknesses, risks, and flawed assumptions.
- Challenge ideas rigorously.
- Do not propose solutions unless explicitly asked.
- Do not soften criticism or hedge excessively.

State awareness:
- You know you are currently in CRITIQUE MODE.
- If asked what mode you are in, answer: "CRITIQUE MODE".
- If asked about your behavior, explain it in terms of CRITIQUE MODE.
"""
}


# -----------------------------
# Model setup
# -----------------------------
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=4096
)

conversation_history: list = []
current_mode: str = "default"

# -----------------------------
# Chat function
# -----------------------------
def chat(user_input: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        *[(msg.type, msg.content) for msg in conversation_history],
        ("human", "{input}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "system_prompt": MODES[current_mode],
        "input": user_input
    })

    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=response.content))

    return f"[{current_mode.upper()} MODE]\n{response.content}"

# -----------------------------
# Mode management
# -----------------------------
def set_mode(mode: str):
    global current_mode
    if mode in MODES:
        current_mode = mode
        print(f"\nMode set to: {mode}\n")
    else:
        print(f"\nInvalid mode. Options: {list(MODES.keys())}\n")

# -----------------------------
# CLI loop
# -----------------------------
def main():
    print("Claude Mode Experiment (LangChain + Anthropic)")
    print(f"Available modes: {list(MODES.keys())}")
    print("Commands:")
    print("  /mode <name>   Switch behavior mode")
    print("  /clear         Clear conversation history")
    print("  /quit          Exit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/quit":
            break

        if user_input == "/clear":
            conversation_history.clear()
            print("\nConversation history cleared.\n")
            continue

        if user_input == "/mode":
            print(f"\nCurrent mode: {current_mode}\n")
            continue

        if user_input.startswith("/mode "):
            set_mode(user_input[6:].strip())
            continue

        response = chat(user_input)
        print(f"\nClaude:\n{response}\n")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
