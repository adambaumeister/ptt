# Convos

Convos (short for "Conversations") is the way this library stores completions prompts. At the time of writing,
most AI providers use a chat style for completions so we mimic that but in a generic way, with additional metadata.

## Convo Files

To simplify the creation of Convo objects, you can define them in Convo formatted files, and use the associated
`ConvoLoader` to build entire, complex prompts using them.

## Jinja2 Loaders

