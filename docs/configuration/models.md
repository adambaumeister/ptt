# AI Model Configuration

Everything in this library relies on the connectivity to AI models to function.

We provide basic wrappers for OpenAI compatible endpoints, as well as vendor specific where it makes sense (such as
the Cloudflare, or Google Vertex APIs).

## Environment Variables

Most of the time, you're just using one model with this package. To simplify this, you can automatically retrieve
the AI model based on the configured AI models.

::: python_to_tools.utils.EnvironmentVariables

## Cloudflare API

::: python_to_tools.ai.cloudflare.client.CloudflareClient.__init__
    options:
        toc_label: "Cloudflare AI Client"
        show_root_toc_entry: false
