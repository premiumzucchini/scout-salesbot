import json
import boto3
import os

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

KB_ID = os.environ["KNOWLEDGE_BASE_ID"]
MODEL_ARN = os.environ["MODEL_ARN"]
FRONTEND_ORIGIN = os.environ["FRONTEND_ORIGIN"]

CORS_HEADERS = {
    "Access-Control-Allow-Origin": FRONTEND_ORIGIN,
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
}

def lambda_handler(event, context):
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
    except Exception:
        return {"statusCode": 400, "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Invalid JSON"})}

    message = body.get("message", "").strip()
    session_id = body.get("sessionId")

    if not message:
        return {"statusCode": 400, "headers": CORS_HEADERS,
                "body": json.dumps({"error": "message is required"})}

    params = {
        "input": {"text": message},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KB_ID,
                "modelArn": MODEL_ARN,
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": (
                            "You are a product support assistant. "
                            "Answer only from the retrieved documentation. "
                            "Never invent features or pricing. "
                            "If the answer is not in the docs say: "
                            "'I couldn't find that in the product docs.'\n\n"
                            "$search_results$\n\nQuestion: $query$"
                        )
                    }
                },
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 5}
                }
            }
        }
    }

    if session_id:
        params["sessionId"] = session_id

    try:
        response = client.retrieve_and_generate(**params)
        answer = response.get("output", {}).get("text", "No answer returned.")
        returned_session_id = response.get("sessionId", "")

        seen = set()
        citations = []
        for citation in response.get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                uri = ref.get("metadata", {}).get("x-amz-bedrock-kb-source-uri", "")
                if uri not in seen:
                    seen.add(uri)
                    citations.append({
                        "title": uri.split("/")[-1] if uri else "Unknown",
                        "uri": uri
                    })

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({
                "answer": answer,
                "sessionId": returned_session_id,
                "citations": citations
            })
        }

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {"statusCode": 502, "headers": CORS_HEADERS,
                "body": json.dumps({"error": str(e)})}
