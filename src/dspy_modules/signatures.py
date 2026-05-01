class TestPlanGenerationSignature(dspy.Signature):
    """
    You are a senior QA engineer.
    Task: Analyze the provided repository code and generate a structured test plan
          for the given scenario in JSON format.
    The test plan must be based ONLY on what is actually implemented in the code context.
    If 'server_url' is provided in the requirement, generate E2E test cases that cover
    both frontend UI behavior and backend API integration scenarios.
    """

    requirement = dspy.InputField(
        desc=(
            "The test scenario prompt from Spring server. "
            "Contains scenario name, description, serial number, attempt count, and guide. "
            "May contain 'base_url' for frontend and 'server_url' for backend API."
        )
    )
    code_context = dspy.InputField(
        desc="Relevant source code chunks retrieved from the repository via RAG."
    )
    reasoning = dspy.OutputField(
        desc="Step-by-step analysis of the repository code relevant to the scenario."
    )
    test_plan = dspy.OutputField(
        desc=(
            "A JSON array of test cases. Each object must have these exact keys: "
            "no (integer), scenario_id (string), scenario_name (string), "
            "description (string), case_id (string), case_name (string), "
            "precondition (string), test_data (string), steps (string), expected_result (string). "
            "Return ONLY the JSON array, no markdown, no backticks."
        )
    )


class TestCodeGenerationSignature(dspy.Signature):
    """
    You are a JavaScript developer specializing in Playwright test automation.
    Task: Given a test plan and repository code context, write a complete
          Playwright test script that covers every test case in the plan.
    """

    test_plan = dspy.InputField(
        desc="JSON array of test cases generated in the previous step."
    )
    code_context = dspy.InputField(
        desc="Relevant source code chunks retrieved from the repository via RAG."
    )

    reasoning = dspy.OutputField(
        desc="Your implementation plan mapping each test case to Playwright code."
    )
    test_code = dspy.OutputField(
        desc=(
            "COMPLETE Playwright test code ONLY. "
            "MUST start with: const { test, expect } = require('@playwright/test'); "
            "MUST use test.describe() with one test() block per test case. "
            "NEVER use chromium.launch() directly. "
            "NEVER use backtick (`) characters — use only single or double quotes. "
            "The test names MUST match case_name values from the test plan. "
            "OUTPUT ONLY VALID JAVASCRIPT CODE. "
            "DO NOT include any explanations, comments outside code, markdown, or reasoning text. "

            "IMPORTANT - Extract URLs from requirement: "
            "If 'base_url' is present, set: const BASE_URL = 'https://...'; "
            "If 'server_url' is present, set: const SERVER_URL = 'https://...'; "
            "ALL page.goto() calls MUST use absolute URLs: BASE_URL + '/path', never '/path' alone. "
            "If both base_url and server_url are present, this is an E2E test. "
            "In E2E mode, verify frontend UI interactions AND backend API responses together. "
            "Use SERVER_URL for direct API calls via page.evaluate() or request context. "

            "IMPORTANT - selector rules for login page: "
            "Login email: '#login-email' "
            "Login password: '#login-password' "
            "Login button: '#btn-login' "
            "Login error message: '#login-error' "
            "Register link: '#go-register' "
            "Register name: '#register-name' "
            "Register email: '#register-email' "
            "Register password: '#register-password' "
            "Register confirm password: '#register-confirm' "
            "Register button: '#btn-register' "
            "Register error: '#register-error' "
            "Register success: '#register-success' "
            "Login link (from register): '#go-login' "
            "Dashboard container: '#dashboard-container' "
            "Logout button: '#btn-logout' "

            "IMPORTANT - Every test() block MUST use try/finally to guarantee screenshot: "
            "Set test.setTimeout(60000) at the top of test.describe. "
            "Use a counter: let _idx = 0; at the top of test.describe. "
            "In each test: _idx++; try { /* test logic */ } finally { "
            "if (page && !page.isClosed()) { "
            "await page.screenshot({ path: 'test-results/screenshot_' + String(_idx).padStart(3,'0') + '.png', fullPage: true }); "
            "} }"
        )
    )
