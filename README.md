# Layr

**Observability for AI agents.** Instrument your agents in a few lines of code and get full visibility into every action, decision, and resource they consume — works with the observability stack you already have.

## Requirements

- Python 3.10+

## Installation

Install the package from PyPI (the distribution name is `layr-sdk`; you import the `layr` module in code):

```bash
pip install layr-sdk
```

## Quickstart

Create an API key from your [Layr account](https://www.getlayr.co), then:

```python
from layr import Agent

agent = Agent(api_key="your-key")

agent.track(
    action="send_email",
    target="user@example.com",
    reasoning="Customer requested order update",
    input_tokens=450,
    output_tokens=210,
    latency_ms=1200,
)
```

## Documentation

Full documentation: [www.getlayr.co/docs](https://www.getlayr.co/docs)

## License

MIT — Built by the Layr team at [www.getlayr.co](https://www.getlayr.co)
