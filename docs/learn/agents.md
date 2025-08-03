# Agents

The most fundamental component of this library is the **Agent**.

Agents are defined by any number of prompts, a name, a description, and finally an AI model.

Agents can be passed *Tools* (functions) to execute, and other agents to pass requests to. In this way, you can 
build a complex tree of agents that are all very good at doing specific things, and also have their own peers to 
forward tasks to when required.

It is important to note that **Agents** do not store their own context. This is the job of their parent 
[Behavior](behavior.md) object.

## Creating an Agent

```python title="Creating an Agent"
from python_to_tools.behavior import Agent
from python_to_tools.utils import JinjaConvoLoader
from python_to_tools.ai.utils import get_model_by_environment_variables
model = get_model_by_environment_variables()

agent = Agent(
    agent_name="default_handler_agent",
    convo_loader=JinjaConvoLoader(
        "single_task_agent.j2",
    ),
    model=model
)
```

## Assigning functions

To assign [Functions](functions.md), it's generally as simple as adding them as tools. 

```python title="Assigning functions to the agent"
def get_weather():
    pass

agent.add_tool(get_weather)
```

Whenever the agent gets passed a request from the user, or from another agent, it can utilize these functions to 
return a result. Functions are very important and are described in more details in the 
[Functions Documentation](functions.md). 

## Adding other agents

You can assign other **Agents** as sub-agents. As soon as you do, the parent agent will always have the ability to
pass a request to another, more suitable agent, if one exists.

```python title="Assigning sub agents"
agent.add_agent(Agent(...))
```

## Getting responses (Resolution)

Within this library, whenever a request is sent to an agent, the act of getting a response is called _Resolution_.

It's called that because an agent may take any of the following steps to generate a response;

1. Simply answer itself by generating text (or images...)
2. Execute a function call and return the result 
3. Pass the request to a subsequent agent

```python title="Resolving text"
result = agent.resolve_from_text("Get me the weather in Sydney, Australia.")

# The weather in Sydney, australia, is 25 degrees!
```