# AI Model Configuration

Everything in this library relies on the connectivity to AI models to function.

We provide basic wrappers for OpenAI compatible endpoints, as well as vendor specific where it makes sense (such as
the Cloudflare, or Google Vertex APIs).

## OpenAI Compatible

You can use this model definition for any OpenAI API compatible endpoint. At the time of writing, this includes 
Vertex AI, OpenAI (obviously) and HuggingFace's text-generation-inference API.

::: python_to_tools.ai.openai.client.OpenAIClient.__init__
    options:
        toc_label: "Cloudflare AI Client"
        show_root_toc_entry: false

## Cloudflare API

::: python_to_tools.ai.cloudflare.client.CloudflareClient.__init__
    options:
        toc_label: "Cloudflare AI Client"
        show_root_toc_entry: false
