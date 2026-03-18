from bedrock_agentcore import BedrockAgentCoreApp
from langchain_core.messages import HumanMessage
from agent import graph

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    prompt = payload.get("prompt", "")
    result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"result": result["messages"][-1].content}

if __name__ == "__main__":
    app.run()
