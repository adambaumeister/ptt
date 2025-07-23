# Python To Tools

![PTT Logo](docs/static/logo.svg)

![GitHub commit activity](https://img.shields.io/github/commit-activity/w/adambaumeister/ptt)
![GitHub License](https://img.shields.io/github/license/adambaumeister/ptt)

## Quickstart

```shell
pip install python-to-tools
```

Check out [this simple example](python_to_tools/examples/my_weather.py) to get started!

## Overview

Python-to-tools (PTT) is a simple, easy to use python library for implementing agentic handling of Python functions.

With PTT, you write Python code, then attach it to AI "Agents". Then you ask those agents to do stuff. Easy!

## Supported Models

 Provider   | Model                                    | Standard Agentic Behavior | Recursive Agentic Behavior 
------------|------------------------------------------|---------------------------|-----------------------------|
 Cloudflare | @cf/meta/llama-3.3-70b-instruct-fp8-fast | ✅                         | ✅                           | 
 Cloudflare | @cf/meta/llama-4-scout-17b-16e-instruct  | ✅                         | ✅                           | 
 Google     | google/gemini-2.5-flash                  | ✅                         | ✅                           | 



## Philosophy

The philosophy of this project is to enable developers to enhance their apps with Agentic AI without requiring a lot of
major code changes or to fully commit to writing a full MCP host.
