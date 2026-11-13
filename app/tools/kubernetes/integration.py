
KUBERNETES_INTEGRATION = {
    "provider_id": "kubernetes",
    "name": "Kubernetes",
    "description": (
        "Kubernetes cluster management, pods, deployments, "
        "services, namespaces, events, logs, and resources."
    ),
    "source_type": "mcp",
    "connection_schema": {
        "fields": [
            {
                "key": "mcp_server_url",
                "label": "Kubernetes MCP Server URL",
                "type": "text",
                "required": True,
                "secret": False,
            },
            {
                "key": "service_account_token",
                "label": "ServiceAccount Bearer Token",
                "type": "password",
                "required": True,
                "secret": True,
            },
            {
                "key": "namespace",
                "label": "Default Namespace (Optional)",
                "type": "text",
                "required": False,
                "secret": False,
            },
        ]
    },
}
