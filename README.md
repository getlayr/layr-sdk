# Layr

Observability for AI agents. Instrument your agents 
in three lines of code and get full visibility into 
every action, decision, and resource they consume — 
works with the observability stack you already have.

## Install

pip install layr-sdk

## Quickstart

from layr import Agent

agent = Agent(api_key="your-key")

agent.track(
    action="send_email",
    target="user@example.com",
    reasoning="Customer requested order update",
    input_tokens=450,
    output_tokens=210,
    latency_ms=1200
)

## Documentation

Full docs at docs.getlayr.io

## License

MIT — Built by the Layr team at getlayr.io
