import dspy

class TestPlanGenerationSignature(dspy.Signature):
    """
    You are a senior QA engineer.
    Task: Analyze the provided repository source code and generate a structured, realistic test plan in JSON format.
    Rules:
    - Base ONLY on components, routes, and functions that actually exist in the code_context.
    - Focus ONLY on the scenario described in requirement. Do NOT overlap with other scenarios.
    - At least 7 test cases generated.
    - Only use loiginId :danimo1 and password:1234 for login test cases, unless requirement specifies otherwise.
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
    - NEVER use page.waitForLoadState('networkidle'). Always use page.waitForLoadState('domcontentloaded') instead to avoid timeout issues.
    - If the scenario requires login, ALWAYS perform login at the start of each test that needs authentication. Use selectors found in code_context.
    - Default test account credentials: loginId='danimo1', password='1234'. Use these unless the requirement specifies otherwise.
    - In loginUser function, ALWAYS add { timeout: 5000 } to all expect assertions to avoid long waits: await expect(page).not.toHaveURL('/login', { timeout: 5000 }).
    - NEVER use waitForTimeout values greater than 1000ms. Replace long waits with waitForLoadState('domcontentloaded').
    - test.setTimeout must be set to 30000 (30 seconds) not 60000, to fail fast.
    - NEVER use expect(pageContent).toBeTruthy() or expect(bodyVisible).toBeTruthy() as the only assertion. Always verify specific UI elements, URLs, or data.
    - After login, ALWAYS verify success by checking the URL does not contain '/login' or by checking a specific authenticated element.
    - After clicking login button, use await page.waitForURL('**/home', { timeout: 10000 }).catch(() => {}) before checking URL to allow redirect to complete.
    - For login page navigation, ALWAYS use BASE_URL + '/' (root URL), NOT BASE_URL + '/login', unless code_context confirms /login route exists. StudyMate login form is on the root page.
    - After filter/sort actions, verify the result by checking element count changes, specific text content, or absence of error pages.
    - After form submission, verify success by checking for success messages, URL changes, or updated data in the DOM.
    - For error cases, verify specific error messages appear using expect(element).toBeVisible() or expect(text).toContain('...').
    - NEVER hardcode year values (e.g. '2026'). Use dynamic checks or regex patterns instead.
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
            "3. Set test.setTimeout(30000) at the top of each test.describe block.\n"
            "4. Use a counter: let _idx = 0; inside test.describe. Increment in each test: _idx++;\n"
            "5. Every test() block MUST have try/finally:\n"
            "   try { /* test logic */ } catch (e) { throw e; } finally {\n"
            "     try {\n"
            "       if (page && !page.isClosed()) {\n"
            "         await page.screenshot({ path: 'test-results/screenshot_' + String(_idx).padStart(3,'0') + '.png', fullPage: true });\n"
            "       }\n"
            "     } catch (screenshotError) {}\n"
            "   }\n"
            "   Even if the test throws an error, the screenshot MUST always be taken in the finally block.\n"
            "6. Use only selectors found in code_context: #id, [name='x'], [placeholder='x'], text='x'.\n"
            "7. Never use Styled-components random class names (e.g. .sc-abc).\n"
            "8. NEVER use expect(pageContent).toBeTruthy() or expect(body).isVisible() as the only assertion.\n"
            "9. Always use meaningful assertions:\n"
            "   - URL check: expect(page.url()).not.toContain('/login')\n"
            "   - Element visibility: await expect(page.locator('selector')).toBeVisible()\n"
            "   - Text content: await expect(page.locator('selector')).toContainText('...')\n"
            "   - Element count: expect(await page.locator('selector').count()).toBeGreaterThan(0)\n"
            "   - No error page: expect(await page.locator('text=500, text=오류').count()).toBe(0)\n"
            "10. For scenarios requiring login, define a reusable loginUser(page) function at the top of test.describe and call it in each test.\n"
            "    Use default credentials: loginId='danimo1', password='1234' unless otherwise specified.\n"
            "11. Output ONLY valid JavaScript code. No markdown, no backticks, no comments outside code."
        )
    )