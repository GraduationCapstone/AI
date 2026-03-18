const { test, expect } = require('@playwright/test');

test('Travel Introduction Page Test', async ({ page }) => {
  // Navigate to the base URL
  await page.goto('https://danimo1.github.io/travel_introduction_page/');

  // Verify the page title
  await expect(page).toHaveTitle('Travel Introduction');

  // Verify the presence of the main heading
  await expect(page.locator('h1')).toHaveText('Welcome to our Travel Introduction Page');

  // Verify the presence of the image gallery
  await expect(page.locator('.image-gallery')).toBeVisible();

  // Verify the presence of the image captions
  await expect(page.locator('.image-caption')).toHaveCount(18);

  // Click on one of the images and verify the modal is displayed
  await page.click('.image-gallery img:first-child');
  await expect(page.locator('.modal')).toBeVisible();

  // Close the modal
  await page.click('.modal .close-button');
  await expect(page.locator('.modal')).toBeHidden();

  // Verify the presence of the 'Learn More' button
  await expect(page.locator('a.learn-more')).toBeVisible();

  // Click the 'Learn More' button and verify the new page is loaded
  await Promise.all([
    page.click('a.learn-more'),
    page.waitForNavigation()
  ]);
  await expect(page).toHaveURL(/.*learn-more/);
});