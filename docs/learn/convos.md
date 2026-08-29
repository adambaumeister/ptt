# Convos

Convos (short for "Conversations") is the way this library stores completions prompts. At the time of writing,
most AI providers use a chat style for completions so we mimic that but in a generic way, with additional metadata.

## Convo Format Example

Convos are simple files that are a human readable representation of an OpenAI API compatible conversation.

```text
""" <----- Triple quotes starts a comment block
This is a comment!
""" <----- ..and end them

[system] <---- square-brackets indicate the role for the next message
You are a smarty pants AI system

[user] 
{{ last_input }}    <---Jinja2 populates the actual user message we want to send, normally!
```

## Convo Files

To simplify the creation of Convo objects, you can define them in Convo formatted files, and use the associated
`ConvoLoader` to build entire, complex prompts using them.

### Jinja2 

The most common and simplest way of defining Convos is via Jinja2 templating. This library ships with
[standard templates](/python_to_tools/templates) for each type of [behavior](behavior.md). But if you need to customize
them, you'll need to write your own!

```text title="Example Convo File using J2 templating to include the message history"
"""
This is a basic agent that is designed to handle a task, optionally calling tools or calling another agent.
"""
[system]
You are an advanced AI agent that is designed to assist users with various tasks. You are a specialist, handling just a single task at a time then moving on. You will be given a series of tools to handle a particular task, and if you cannot handle the task, you should call the `next_agent` tool.

You cannot ask the user to do things for you. If you have failed to solve the task, you should communicate that to the user.

{% if context %}
{% for msg in context.messages %}
[{{ msg.role }}]
{{ msg.content }}
{% endfor %}
{% endif %}

[user]
Complete the following task, then return your results. If you cannot complete the task, return FAIL, followed by
a reason for the failure.

Task: {{ last_input }}
```


