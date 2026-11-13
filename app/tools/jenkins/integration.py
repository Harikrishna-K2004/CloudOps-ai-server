
JENKINS_INTEGRATION = {
    "provider_id": "jenkins",
    "name": "Jenkins",
    "description": (
        "Jenkins CI/CD pipelines, jobs, builds, "
        "build logs, and automation."
    ),
    "source_type": "mcp",
    "connection_schema": {
        "fields": [
            {
                "key": "jenkins_url",
                "label": "Jenkins URL",
                "type": "text",
                "required": True,
                "secret": False,
            },
            {
                "key": "username",
                "label": "Jenkins Username",
                "type": "text",
                "required": True,
                "secret": False,
            },
            {
                "key": "api_token",
                "label": "Jenkins API Token",
                "type": "password",
                "required": True,
                "secret": True,
            },
        ]
    },
}
