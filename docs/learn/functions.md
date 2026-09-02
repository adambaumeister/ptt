# Functions and Tools

Function/Tool calling is the ultimate purpose of this library. Agentic handling is nothing if it can't actually *do*
stuff, after all!

PTT tries to make it as easy as possible to integrate your python functions into an agentic flow by simply
passing them to agents. You only have to add some additional annotations to the arguments to make it work.

## Creating a basic Function

Let's create a basic function that takes two numbers and adds them together.

```python title="Basic, agent compatible function"
from typing import Annotated
from python_to_tools.ai.generic_models import ToolParameter
def add_two_numbers(
        x: Annotated[
            str, ToolParameter(type="string", description="The first number")
        ],
        y: Annotated[
            str, ToolParameter(type="string", description="The second number")
        ]
) -> str:
    """Adds two numbers, returning their result"""
    return str(x+y)
```

When we add this to an [Agent](agents.md#assigning-functions), it will be converted into an OpenAI API Schema compatible
function call using the following logic:

 * The function __doc___, or docstring, is what's used as the function description.
 * Each argument will be added as arguments with the associated type, description, and name
 * For any argument that does NOT have a default value, it will be marked as 'required'

## Understanding return data types

Essentially, any function that an agent can use must ultimately return a string. One of the greatest contributions to
modern software development that AI has made has taken us back to a simpler, easier time, before people realized
"type safety" was a good idea. 

This library provides some cheats, however, as models have gotten much better at understanding programmatic or 
serialized data.

You can return normal data structures like dicts or lists;

```python title="Returning formatted data"
def get_my_location() -> dict:
    """Gets the user's location"""
    return {"city": "sydney", "country": "australia"}
```

or functions that return [Pydantic Models](https://docs.pydantic.dev/latest/)

```python title="Returning formatted data"
from pydantic import BaseModel
class Location(BaseModel):
    city: str
    country: str
    
def get_my_location() -> Location:
    """Get's the user's location"""
    return Location(city="sydney", country="Australia")
```

As you might imagine, in this cases the data simply gets serialized as JSON before being passed to the model.