import dspy

class TestPlanGenerationSignature(dspy.Signature):
    """
    You are a senior QA engineer.
    Task: Analyze the provided repository source code and generate a structured, realistic test plan in JSON format.
    Rules:
    - Base ONLY on components, routes, and functions that actually exist in the code_context.
    - Focus ONLY on the scenario described in requirement. Do NOT overlap with other scenarios.
    - If base_url is provided in requirement, use it as the target URL for test cases.
    - If server_url is provided in requirement, include E2E test cases that verify both frontend UI and backend API responses.
    - Output ONLY a raw JSON array. No markdown, no backticks, no explanation.
    """

    requirement = dspy.InputField(
        desc="Test scenario description. May contain base_url (frontend URL) and server_url (backend API URL)."
    )
    code_context = dspy.InputField(
        desc="Source code chunks retrieved via RAG (React components, API functions, routes, etc)."
    )
    reasoning = dspy.OutputField(
        desc="Step-by-step analysis: which components, selectors (id, class, placeholder) were found in the code."
    )
    scenario_serial = dspy.InputField(
        desc="2-digit scenario serial number provided by backend (e.g. '02' for login). Use this to construct IDs."
    )
    scenario_attempt = dspy.InputField(
        desc="2-digit attempt number provided by backend (e.g. '01'). Use this to construct IDs."
    )
    test_plan = dspy.OutputField(
        desc=(
            "A raw JSON array of test cases. Each object must have exactly these keys: "
            "no (integer), scenario_id (string), scenario_name (string), description (string), "
            "case_id (string), case_name (string), precondition (string), test_data (string), "
            "steps (string), expected_result (string). "
            "CRITICAL ID FORMAT RULES: "
            "- scenario_id MUST be: T{scenario_serial}{scenario_attempt} (e.g. T0201 if serial=02, attempt=01). "
            "- case_id MUST be: T{scenario_serial}{scenario_attempt}_{nn} where nn is 2-digit sequential number (e.g. T0201_01, T0201_02). "
            "- All test cases in the same scenario share the same scenario_id. "
            "Output ONLY the JSON array. No markdown, no backticks, no extra text."
        )
    )


class TestCodeGenerationSignature(dspy.Signature):
    """
    You are a Playwright automation expert.
    Task: Write complete, executable Playwright test code in JavaScript (CommonJS) based on the test plan.
    Rules:
    - MUST start with: const { test, expect } = require('@playwright/test');
    - NEVER use TypeScript syntax (no import, no type annotations, no interface, no : Type).
    - Use actual selectors found in code_context (id, name, placeholder, text). Never use random class names.
    - If base_url is in the plan, define: const BASE_URL = 'https://...'; and use it in all page.goto() calls.
    - If server_url is in the plan, define: const SERVER_URL = 'https://...'; and use page.request or fetch for API verification.
    - ALL page.goto() calls MUST use absolute URLs. Never use relative paths like '/path'.
    - Every test() block MUST use try/finally to guarantee screenshot.
    - Output ONLY pure JavaScript code. No markdown, no backticks, no explanation.
    """

    test_plan_item = dspy.InputField(
        desc="JSON array of test cases from the test plan."
    )
    code_context = dspy.InputField(
        desc="Source code chunks relevant to the feature being tested."
    )

    generated_code = dspy.OutputField(
        desc=(
            "Complete Playwright JavaScript (CommonJS) test code. Strict rules:\n"
            "1. First line MUST be: const { test, expect } = require('@playwright/test');\n"
            "2. NEVER use TypeScript: no import statements, no type annotations (page: Page), no interfaces.\n"
            "3. Set test.setTimeout(60000) at the top of each test.describe block.\n"
            "4. Use a counter: let _idx = 0; inside test.describe. Increment in each test: _idx++;\n"
            "5. Every test() block MUST have try/finally:\n"
            "   try { /* test logic */ } finally {\n"
            "     if (page && !page.isClosed()) {\n"
            "       await page.screenshot({ path: 'test-results/screenshot_' + String(_idx).padStart(3,'0') + '.png', fullPage: true });\n"
            "     }\n"
            "   }\n"
            "6. Use only selectors found in code_context: #id, [name='x'], [placeholder='x'], text='x'.\n"
            "7. Never use Styled-components random class names (e.g. .sc-abc).\n"
            "8. Output ONLY valid JavaScript code. No markdown, no backticks, no comments outside code."
        )
    )