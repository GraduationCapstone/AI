const { test, expect } = require('@playwright/test');

test('Travel Introduction Page Tests', async ({ page }) => {
  // 1. Verify the page loads successfully
  await page.goto('https://danimo1.github.io/travel_introduction_page/');
  await expect(page).toHaveURL('https://danimo1.github.io/travel_introduction_page/');

  // 2. Verify the page title matches the expected title
  await expect(page).toHaveTitle('Travel Introduction Page');

  // 3. Verify the presence of the main navigation menu
  await expect(page.locator('nav')).toBeVisible();
  await expect(page.locator('nav a')).toHaveCount(3);
  await expect(page.locator('nav a')).toContainText(['Home', 'Destinations', 'About']);

  // 4. Verify the presence and content of the hero section
  await expect(page.locator('.hero')).toBeVisible();
  await expect(page.locator('.hero h1')).toContainText('Explore the World');
  await expect(page.locator('.hero p')).toContainText('Discover the beauty of different destinations');

  // 5. Verify the presence and content of the destination sections
  await expect(page.locator('.destination')).toHaveCount(3);
  await expect(page.locator('.destination h2')).toContainText(['London', 'Paris', 'Brussels']);
  await expect(page.locator('.destination p')).toContainText(['Explore the rich history and culture of London.', 'Experience the charm and romance of Paris.', 'Indulge in the delights of Brussels.']);

  // 6. Verify the presence and content of the footer
  await expect(page.locator('footer')).toBeVisible();
  await expect(page.locator('footer p')).toContainText('© 2023 Travel Introduction Page. All rights reserved.');
});