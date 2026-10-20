from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_aws_infrastructure_has_cost_guardrails():
    template = (PROJECT_ROOT / "infra" / "aws" / "cloudformation.yml").read_text()

    assert "AllowedValues: [0, 1]" in template
    assert "Value: disabled" in template
    assert "RetentionInDays: 7" in template
    assert "MAX_CONCURRENT_MATCHES" in template
    assert 'Cpu: "1024"' in template
    assert 'Memory: "2048"' in template
    assert "MonthlyCostBudget" in template
    assert "AnalyticsDashboard" in template
    assert "CacheBehaviors:" in template
    assert "PathPattern: /api/*" in template
    assert "PathPattern: /api/celebrity-photo/*" in template
    assert "DefaultTTL: 300" in template
    assert "DefaultTTL: 86400" in template
