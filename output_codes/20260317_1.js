const { test, expect } = require('@playwright/test');

test('Travel Introduction Page Test', async ({ page }) => {
  // Navigate to the base URL
  await page.goto('https://danimo1.github.io/travel_introduction_page/');

  // Verify the page title
  await expect(page).toHaveTitle('Travel Introduction Page');

  // Verify the presence of specific elements
  await expect(page.locator('h1')).toBeVisible();
  await expect(page.locator('p')).toBeVisible();
  await expect(page.locator('img')).toHaveCount(18);

  // Verify the behavior of interactive elements (if any)
  // For example, if there are links or buttons, you can add tests to click on them and verify the expected behavior
})