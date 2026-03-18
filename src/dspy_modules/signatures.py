import dspy


class TestPlanGenerationSignature(dspy.Signature):
    """
    You are a senior QA engineer.
    Task: Analyze the provided repository code and generate a structured test plan
          for the given scenario in JSON format.
    The test plan must be based ONLY on what is actually implemented in the code context.
    """

    requirement = dspy.InputField(
        desc=(
            "The test scenario prompt from Spring server. "
            "Contains scenario name, description, serial number, attempt count, and guide."
        )
    )
    code_context = dspy.InputField(
        desc="Relevant source code chunks retrieved from the repository via RAG."
    )
    base_url = dspy.InputField(
        desc="The base URL of the application under test."
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
    base_url = dspy.InputField(
        desc="The base URL of the application under test."
    )

    reasoning = dspy.OutputField(
        desc="Your implementation plan mapping each test case to Playwright code."
    )
    test_code = dspy.OutputField(
        desc=(
            "COMPLETE Playwright test code. "
            "MUST start with: const { test, expect } = require('@playwright/test'); "
            "MUST use test.describe() with one test() block per test case. "
            "NEVER use chromium.launch() directly. "
            "NEVER use backtick (`) characters — use only single or double quotes. "
            "The test names MUST match case_name values from the test plan."
        )
    )