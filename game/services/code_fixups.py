import re


def _fix_mock_name_pattern(code: str) -> str:
    """
    Rewrite MagicMock(name='foo') → MagicMock() with explicit .name assignment.
    This is a known LLM failure mode — the name= constructor arg sets the mock's
    internal identifier, not the .name attribute.
    """
    pattern = re.compile(r"(\s*)(\w+)\s*=\s*MagicMock\(name=['\"]([^'\"]+)['\"]\)")

    def rewrite(m: re.Match) -> str:
        indent, varname, name_value = m.group(1), m.group(2), m.group(3)
        return (
            f"{indent}{varname} = MagicMock()\n{indent}{varname}.name = '{name_value}'"
        )

    return pattern.sub(rewrite, code)
