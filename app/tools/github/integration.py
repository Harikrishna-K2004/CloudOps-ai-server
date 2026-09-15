GITHUB_INTEGRATION = {
    "provider_id": "github",
    "name": "GitHub",
    "description": "GitHub repositories, commits, pull requests, issues, and actions.",
    "source_type": "mcp",
    "connection_schema": {
        "fields": [
            {
                "key": "personal_access_token",
                "label": "Personal Access Token",
                "type": "password",
                "required": True,
                "secret": True,
            }
        ]
    },
}