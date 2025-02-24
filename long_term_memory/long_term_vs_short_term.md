# Long term vs short term memory

## What is memory?
- Memory is a cognitive function that allows people to store, retrieve and use information to understand their present and future.

### Within session (thread) memory
- In context of AI application, thread in langgraph store the converstaion history between AI bot and user within a single chat session. User can come back to that chat session at later point in time. The thread contain entire history.

- Scope -> With session (thread)
- Example use-case -> Persist conversational history, allow interruptions in a chat (e.g., if user is idle or to all human-in-loop)
- LangGraph usage -> Checkpointer


### Across session (thread) memory
- Some information can be persisted across all session with that user. It could be todos, user profile. We want this information for  across all the session.

- Scope -> Across session (thread)
- Example use-case -> Remember information about a specific user across all chat sessions
- LangGraph usage -> Store


## Memory of AI bot
- Semantic (Facts) -> Facts about a user
- Episodic (Memories) -> Past agent actions
- Procedural (Instructions) -> Agent's system prompts

### How can we store facts?
- Using user profile. 